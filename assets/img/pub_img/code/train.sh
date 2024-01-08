curr_dir=`pwd`
ulimit -n 6000
set -e

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

# Variables to change
prefix="wsp-nmt_escher_muse" # Naming prefix
test_langs=(es_XX fr_XX it_IT ro_RO pt_XX)

### !!! UNCOMMENT THIS WITH YOUR OWN PATHS !!! ###

# More or less fixed
# base_dir=<path_to_base_dir> # Path to the root directory where you want processed files to be stored
# datasets_dir=<path_to_datasets_dir> # Path to the directory where the raw datasets are stored
# FAIRSEQ=<path_to_fairseq_dir> # Path to the directory where the fairseq scripts are stored
# SPM=<path_to_spm_dir> # Path to the directory where the sentencepiece scripts are stored
# training_config=<path_to_training_config> # Path to the training config file
# mcolt_dir=<path_to_mcolt_dir> # Path to the directory where the mcolt (https://github.com/PANXiao1994/mRASP2/tree/master/mcolt) scripts are stored

# Derived
data_dir="${base_dir}/${prefix}/data"
checkpoints_dir="${base_dir}/${prefix}/checkpoints"
results_dir="${base_dir}/${prefix}/results"
dict_dir="${data_dir}/dict"
spm_dir="${data_dir}/spm"

source ${base_dir}/load_config.sh ${training_config} ${base_dir} # We use https://github.com/PANXiao1994/mRASP2/blob/master/scripts/load_config.sh to load the training config file

mkdir -p ${checkpoints_dir} ${results_dir}

if [[ ! -f "${results_dir}/training.done" ]];
then
    python ${FAIRSEQ}/train.py ${data_dir} \
        --user-dir ${mcolt_dir} \
        --save-dir ${checkpoints_dir} \
        --mono-data ${data_dir}/mono \
        ${options} --do_shuf --patience 10 \
        --seed ${seed} --ddp-backend no_c10d 1>&2
fi

touch "${results_dir}/training.done"

SPM_MODEL=${spm_dir}/${prefix}.model
MODEL="${checkpoints_dir}/checkpoint_best.pt"

if [ ! -f "${dict_dir}/dict.src.txt" ] || [ ! -f "${SPM_MODEL}" ] || [ ! -f "${MODEL}" ] ; then
    echo "[ERROR]: Prefix ${prefix} does not contain spm model or dictionary or trained model. Exiting...";
    exit 1
fi

# Hypothesis generation for WMT/FLORES test sets
SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

for TGT_LANG in "${test_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    echo "Target lang: ${TGT_LANG_SHORT}"

    PARALLEL_DIR="${datasets_dir}/parallel/en-${TGT_LANG_SHORT}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/test.detok"
    TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
    lang_pair="${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"

    rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC_LANG} > ${TEST_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT_LANG} > ${TEST_FINAL}.spm.${TGT_LANG_SHORT}
    
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TEST_FINAL}.spm.${SRC_LANG_SHORT} > ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT}
	cp ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}

    echo "Tokenizing Done"
    python ${FAIRSEQ}/preprocess.py \
            --source-lang ${SRC_LANG_SHORT} --target-lang  ${TGT_LANG_SHORT} \
            --testpref ${TEST_FINAL}.prefixed.spm \
            --srcdict ${dict_dir}/dict.src.txt \
            --tgtdict ${dict_dir}/dict.src.txt \
            --destdir ${data_dir}/test/${lang_pair}/ \
            --thresholdtgt 0 --thresholdsrc 0 --seed ${seed} \
            --amp --workers 72

    rm -rf ${TEST_FINAL}.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}
    echo "Binarizing Done"

    for i in {0..3};
    do
        mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${mcolt_dir} \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} --seed ${seed} \
            --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${TGT_LANG_SHORT} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 4 --distributed-rank ${i} \
            --results-path ${results_dir}/${lang_pair}/shard_id${i} &
    done
    wait

    echo "Translation done"
    cat ${results_dir}/${lang_pair}/shard_id*/*.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d" " -f2- > ${results_dir}/${lang_pair}/output.tok.txt
    ${SPM}/spm_decode --model=${SPM_MODEL} < ${results_dir}/${lang_pair}/output.tok.txt > ${results_dir}/${lang_pair}/output.detok.txt
    rm -r ${results_dir}/${lang_pair}/output.tok.txt ${data_dir}/test/${lang_pair}
    rm -r ${results_dir}/${lang_pair}/shard_id*/*.txt 

    echo "For lang pair: ${lang_pair} Model type: ${prefix}, BLEU is:"
    mkdir -p ${results_dir}/json/
    sacrebleu -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf.json
        sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf_spm.json
    comet-score -s "${TEST_PREFIX}.${SRC_LANG}" -r "${TEST_PREFIX}.${TGT_LANG}" -t ${results_dir}/${lang_pair}/output.detok.txt --gpus 1 --to_json true --model_storage_path /bask/projects/x/xngs6460-languages/viyer/.cache  > ${results_dir}/json/${lang_pair}_comet.json
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"
TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"

for SRC_LANG in "${test_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    echo "Source lang: ${SRC_LANG_SHORT}"

    PARALLEL_DIR="${datasets_dir}/parallel/en-${SRC_LANG_SHORT}-wmt"
    TEST_PREFIX="${PARALLEL_DIR}/test.detok"
    TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
    lang_pair="${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"

    rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC_LANG} > ${TEST_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT_LANG} > ${TEST_FINAL}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TEST_FINAL}.spm.${SRC_LANG_SHORT} > ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT}
	cp ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}

    echo "Tokenizing Done"
    python ${FAIRSEQ}/preprocess.py \
            --source-lang ${SRC_LANG_SHORT} --target-lang  ${TGT_LANG_SHORT} \
            --testpref ${TEST_FINAL}.prefixed.spm \
            --srcdict ${dict_dir}/dict.src.txt \
            --tgtdict ${dict_dir}/dict.src.txt \
            --destdir ${data_dir}/test/${lang_pair}/ \
            --thresholdtgt 0 --thresholdsrc 0 --seed ${seed} \
            --amp --workers 72

    rm -rf ${TEST_FINAL}.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.spm.${TGT_LANG_SHORT}
    echo "Binarizing Done"

    for i in {0..3};
    do
        mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${mcolt_dir} \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} --seed ${seed} \
            --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${TGT_LANG_SHORT} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 4 --distributed-rank ${i} \
            --results-path ${results_dir}/${lang_pair}/shard_id${i} &
    done
    wait

    echo "Translation done"
    cat ${results_dir}/${lang_pair}/shard_id*/*.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d" " -f2- > ${results_dir}/${lang_pair}/output.tok.txt
    ${SPM}/spm_decode --model=${SPM_MODEL} < ${results_dir}/${lang_pair}/output.tok.txt > ${results_dir}/${lang_pair}/output.detok.txt
    rm -r ${results_dir}/${lang_pair}/output.tok.txt ${data_dir}/test/${lang_pair}
    rm -r ${results_dir}/${lang_pair}/shard_id*/*.txt 

    echo "For lang pair: ${lang_pair} Model type: ${prefix}, BLEU is:"
    mkdir -p ${results_dir}/json/
    sacrebleu -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf.json
        sacrebleu --tokenize spm -m bleu chrf --chrf-word-order 2 "${TEST_PREFIX}.${TGT_LANG}" < ${results_dir}/${lang_pair}/output.detok.txt > ${results_dir}/json/${lang_pair}_bleu_chrf_spm.json
    comet-score -s "${TEST_PREFIX}.${SRC_LANG}" -r "${TEST_PREFIX}.${TGT_LANG}" -t ${results_dir}/${lang_pair}/output.detok.txt --gpus 1 --to_json true --model_storage_path /bask/projects/x/xngs6460-languages/viyer/.cache  > ${results_dir}/json/${lang_pair}_comet.json
done

# Hypothesis generation for DiBiMT
test_langs=(es_XX it_IT)

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

for TGT_LANG in "${test_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    echo "Target lang: ${TGT_LANG_SHORT}"

    PARALLEL_DIR="${datasets_dir}/en-${TGT_LANG_SHORT}-dibimt"
    prefix="${prefix}_dibimt"
    TEST_PREFIX="${PARALLEL_DIR}/test.detok"
    TEST_FINAL="${PARALLEL_DIR}/preprocessed/${prefix}/test.detok"
    lang_pair="${SRC_LANG_SHORT}-${TGT_LANG_SHORT}_dibimt"

    rm -rf ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/
    mkdir -p ${PARALLEL_DIR}/preprocessed/${prefix}/ ${data_dir}/test/${lang_pair}/

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${SRC_LANG} > ${TEST_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TEST_PREFIX}.${TGT_LANG} > ${TEST_FINAL}.spm.${TGT_LANG_SHORT}
    
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TEST_FINAL}.spm.${SRC_LANG_SHORT} > ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT}
	cp ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}

    echo "Tokenizing Done"
    python ${FAIRSEQ}/preprocess.py \
            --source-lang ${SRC_LANG_SHORT} --target-lang  ${TGT_LANG_SHORT} \
            --testpref ${TEST_FINAL}.prefixed.spm \
            --srcdict ${dict_dir}/dict.src.txt \
            --tgtdict ${dict_dir}/dict.src.txt \
            --destdir ${data_dir}/test/${lang_pair}/ \
            --thresholdtgt 0 --thresholdsrc 0 --seed ${seed} \
            --amp --workers 128

    rm -rf ${TEST_FINAL}.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.spm.${TGT_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${SRC_LANG_SHORT} ${TEST_FINAL}.prefixed.spm.${TGT_LANG_SHORT}
    echo "Binarizing Done"

    for i in {0..3};
    do
        mkdir -p ${results_dir}/${lang_pair}/shard_id${i}
        CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${data_dir}/test/${lang_pair} \
            --max-tokens 4000 \
            --user-dir ${mcolt_dir} \
            --skip-invalid-size-inputs-valid-test \
            --max-source-positions 4000 \
            --max-target-positions 4000 \
            --path ${MODEL} --seed ${seed} \
            --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${TGT_LANG_SHORT} " | tr '[a-z]' '[A-Z]'` \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size 4 --distributed-rank ${i} \
            --results-path ${results_dir}/${lang_pair}/shard_id${i} &
    done
    wait


    echo "Translation done"
    cat ${results_dir}/${lang_pair}/shard_id*/*.txt | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d" " -f2- > ${results_dir}/${lang_pair}/output.tok.txt
    ${SPM}/spm_decode --model=${SPM_MODEL} < ${results_dir}/${lang_pair}/output.tok.txt > ${results_dir}/${lang_pair}/output.detok.txt
    rm -r ${results_dir}/${lang_pair}/output.tok.txt ${data_dir}/test/${lang_pair}
    rm -r ${results_dir}/${lang_pair}/shard_id*/*.txt 
done

touch "${results_dir}/dibimt.done"