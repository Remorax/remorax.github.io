set -e

# Variables to change
prefix="wsp-nmt_escher_muse" # Naming prefix
train_langs=(es_XX fr_XX it_IT ro_RO)
mono_langs=(es_XX fr_XX it_IT ro_RO en_XX pt_XX)
all_langs=(en es fr pt it ro)

seed="0"
rr="0.1"

### !!! UNCOMMENT THIS WITH YOUR OWN PATHS !!! ###

# More or less fixed
# base_dir=<path_to_base_dir> # Path to the root directory where you want processed files to be stored
# datasets_dir=<path_to_datasets_dir> # Path to the directory where the raw datasets are stored
# preprocessing_dir=<path_to_preprocessing_dir> # Path to the directory where the preprocessing scripts are stored
# moses_dir=<path_to_moses_dir> # Path to the directory where the moses scripts are stored
# FAIRSEQ=<path_to_fairseq_dir> # Path to the directory where the fairseq scripts are stored
# SPM=<path_to_spm_dir> # Path to the directory where the sentencepiece scripts are stored
# MUSE_DIR=<path_to_muse_dir> # Path to the directory where the MUSE dictionaries are stored

# Derived
data_dir="${base_dir}/${prefix}/data"
checkpoints_dir="${base_dir}/${prefix}/checkpoints"
results_dir="${base_dir}/${prefix}/results"
dict_dir="${data_dir}/dict"
spm_dir="${data_dir}/spm"
spm_corpus="${data_dir}/spm_corpus"
MODEL=${spm_dir}/${prefix}.model

TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
MONO_SRC="${data_dir}/mono/train.src"
MONO_TGT="${data_dir}/mono/train.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

rm -rf ${base_dir}/${prefix}
mkdir -p ${base_dir}/${prefix} ${data_dir} ${checkpoints_dir} ${results_dir} ${dict_dir} ${data_dir}/mono  ${spm_dir}

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/parallel/en-${lang}"
    
    WSD_RESULTS="${PARALLEL_DIR}/escher-${SRC_LANG_CAPS}.pkl" # Path to the WSD results (Substitute with AMuSE-WSD results if needed)
    BN_TRANSLATIONS="${PARALLEL_DIR}/translations-${SRC_LANG_CAPS}.pkl" # Path to the BabelNet translations
    SRC_TOK="${base_dir}/${prefix}/en-${lang}/train.tok.${SRC_LANG_SHORT}-${lang}.${SRC_LANG_SHORT}"
    TGT_TOK="${base_dir}/${prefix}/en-${lang}/train.tok.${SRC_LANG_SHORT}-${lang}.${lang}"
    SRC_FINAL="${base_dir}/${prefix}/en-${lang}/train.${SRC_LANG_SHORT}-${lang}.${SRC_LANG_SHORT}"
    TGT_FINAL="${base_dir}/${prefix}/en-${lang}/train.${SRC_LANG_SHORT}-${lang}.${lang}"

    echo "[STATUS] Codeswitching for the En->${lang} direction..."
    mkdir -p ${base_dir}/${prefix}/en-${lang}/
    # Codeswitch the corpus
    python ${preprocessing_dir}/codeswitch.py \
        --disambiguated_corpus ${WSD_RESULTS} \
        --bn_translations ${BN_TRANSLATIONS} \
        --src_lang ${SRC_LANG_CAPS} \
        --save_cs_path ${SRC_TOK} \
        --save_tgt_path ${TGT_TOK} \
        --replacement_ratio ${rr} \
        --exclude_propn --seed ${seed} \
        --use_muse --muse_dir ${MUSE_DIR}
    echo "Done"

    perl ${moses_dir}/moses_detokenizer.perl -l ${SRC_LANG_SHORT} < ${SRC_TOK} > ${SRC_FINAL}
    cp ${TGT_TOK} ${TGT_FINAL}
done

TGT_LANG="en_XX"
TGT_LANG_SHORT="${TGT_LANG:0:2}"

for SRC_LANG in "${train_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${lang^^}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/parallel/en-${lang}"
    
    WSD_RESULTS="${PARALLEL_DIR}/escher-${SRC_LANG_CAPS}.pkl" # Path to the WSD results (Substitute with AMuSE-WSD results if needed)
    BN_TRANSLATIONS="${PARALLEL_DIR}/translations-${SRC_LANG_CAPS}.pkl" # Path to the BabelNet translations
    SRC_TOK="${base_dir}/${prefix}/en-${lang}/train.tok.${lang}-${TGT_LANG_SHORT}.${lang}"
    TGT_TOK="${base_dir}/${prefix}/en-${lang}/train.tok.${lang}-${TGT_LANG_SHORT}.${TGT_LANG_SHORT}"
    SRC_FINAL="${base_dir}/${prefix}/en-${lang}/train.${lang}-${TGT_LANG_SHORT}.${lang}"
    TGT_FINAL="${base_dir}/${prefix}/en-${lang}/train.${lang}-${TGT_LANG_SHORT}.${TGT_LANG_SHORT}"

    echo "[STATUS] Codeswitching for the ${lang}->En direction..."
    mkdir -p ${base_dir}/${prefix}/en-${lang}/
    # Codeswitch the corpus
    python ${preprocessing_dir}/codeswitch.py \
        --disambiguated_corpus ${WSD_RESULTS} \
        --bn_translations ${BN_TRANSLATIONS} \
        --src_lang ${SRC_LANG_CAPS} \
        --save_cs_path ${SRC_TOK} \
        --save_tgt_path ${TGT_TOK} \
        --replacement_ratio ${rr} \
        --exclude_propn --seed ${seed} \
        --use_muse --muse_dir ${MUSE_DIR}
    echo "Done"

    # Detokenize word-tokenized code-switched corpora
    perl ${moses_dir}/moses_detokenizer.perl -l ${SRC_LANG_SHORT} < ${SRC_TOK} > ${SRC_FINAL}
    cp ${TGT_TOK} ${TGT_FINAL}
done


for SRC_LANG in "${mono_langs[@]}"
do
    SRC_LANG_SHORT="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

    MONO_DIR="${datasets_dir}/mono"
    WSD_RESULTS="${MONO_DIR}/escher-${SRC_LANG_CAPS}.pkl" # Path to the WSD results (Substitute with AMuSE-WSD results if needed)
    BN_TRANSLATIONS="${MONO_DIR}/translations-${SRC_LANG_CAPS}.pkl" # Path to the BabelNet translations
    SRC_TOK="${base_dir}/${prefix}/mono-${SRC_LANG_SHORT}/train.tok.mono${SRC_LANG_SHORT}noised"
    TGT_TOK="${base_dir}/${prefix}/mono-${SRC_LANG_SHORT}/train.tok.mono${SRC_LANG_SHORT}target"
    SRC_FINAL="${base_dir}/${prefix}/mono-${SRC_LANG_SHORT}/train.mono${SRC_LANG_SHORT}noised"
    TGT_FINAL="${base_dir}/${prefix}/mono-${SRC_LANG_SHORT}/train.mono${SRC_LANG_SHORT}target"

    echo "[STATUS] Codeswitching the ${SRC_LANG_SHORT} mono corpus..."

    mkdir -p ${base_dir}/${prefix}/mono-${SRC_LANG_SHORT}

    # Codeswitch the corpus
    python ${preprocessing_dir}/codeswitch.py \
        --disambiguated_corpus ${WSD_RESULTS} \
        --bn_translations ${BN_TRANSLATIONS} \
        --src_lang ${SRC_LANG_CAPS} \
        --save_cs_path ${SRC_TOK} \
        --save_tgt_path ${TGT_TOK} \
        --replacement_ratio ${rr} \
        --exclude_propn --seed ${seed} \
        --use_muse --muse_dir ${MUSE_DIR}
    echo "Done"

    # Detokenize word-tokenized code-switched corpora
    perl ${moses_dir}/moses_detokenizer.perl -l ${SRC_LANG_SHORT} < ${SRC_TOK} > ${SRC_FINAL}
    cp ${TGT_TOK} ${TGT_FINAL}
done

rm -rf ${spm_corpus}
touch $spm_corpus

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    PARALLEL_DIR="${datasets_dir}/en-${lang}"
    SRC_FINAL="${base_dir}/${prefix}/en-${lang}/train.${SRC_LANG_SHORT}-${lang}.${SRC_LANG_SHORT}"
    TGT_FINAL="${base_dir}/${prefix}/en-${lang}/train.${SRC_LANG_SHORT}-${lang}.${lang}"

    cat ${SRC_FINAL} ${TGT_FINAL} >> $spm_corpus

    SRC_FINAL="${base_dir}/${prefix}/en-${lang}/train.${lang}-${SRC_LANG_SHORT}.${SRC_LANG_SHORT}"
    TGT_FINAL="${base_dir}/${prefix}/en-${lang}/train.${lang}-${SRC_LANG_SHORT}.${lang}"

    cat ${SRC_FINAL} ${TGT_FINAL} >> $spm_corpus

done

for TGT_LANG in "${mono_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    MONO_DIR="${datasets_dir}/mono-${lang}"
    MONO_FILE="${base_dir}/${prefix}/mono-${lang}/train"

    cat "${MONO_FILE}.mono${lang}noised" "${MONO_FILE}.mono${lang}target" >> $spm_corpus
done

shuf $spm_corpus > "${spm_corpus}.shuf"

echo "Training an spm model"
${SPM}/spm_train --input="${spm_corpus}.shuf" --train_extremely_large_corpus=true --model_prefix="${spm_dir}/${prefix}" --vocab_size=32000 --character_coverage=1.0  --model_type=unigram --input_sentence_size=20000000 --shuffle_input_sentence=true
${SPM}/spm_encode --model="${MODEL}" < ${spm_corpus}.shuf > "${spm_corpus}.spm.src"

python ${FAIRSEQ}/preprocess.py \
        --source-lang src --target-lang src \
        --trainpref "${spm_corpus}.spm" \
        --destdir ${dict_dir} --only-source \
        --thresholdtgt 0 --thresholdsrc 0 \
        --dict-only --joined-dictionary  \
        --workers 32 --seed ${seed}

for lang in "${all_langs[@]}"
do
    lang_caps="${lang^^}"
    echo -e "LANG_TOK_${lang_caps} 1" >> "${dict_dir}/dict.src.txt"
done


TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
MONO_SRC="${data_dir}/mono/train.src"
MONO_TGT="${data_dir}/mono/train.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

for TGT_LANG in "${train_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    PARALLEL_DIR="${datasets_dir}/parallel/en-${TGT_LANG_SHORT}"

    TRAIN_PREFIX="${base_dir}/${prefix}/en-${TGT_LANG_SHORT}/train.${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"
    TRAIN_PREFIX_REV="${base_dir}/${prefix}/en-${TGT_LANG_SHORT}/train.${TGT_LANG_SHORT}-${SRC_LANG_SHORT}"
    DEV_PREFIX="${PARALLEL_DIR}/Europarl.cleaned.dev.detok"
    DEV_FINAL="${base_dir}/${prefix}/en-${TGT_LANG_SHORT}/dev"
    
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${SRC_LANG_SHORT} > ${TRAIN_PREFIX}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${TGT_LANG_SHORT} > ${TRAIN_PREFIX}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX_REV}.${SRC_LANG_SHORT} > ${TRAIN_PREFIX_REV}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX_REV}.${TGT_LANG_SHORT} > ${TRAIN_PREFIX_REV}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${SRC_LANG} > ${DEV_FINAL}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${TGT_LANG} > ${DEV_FINAL}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_PREFIX}.spm.${SRC_LANG_SHORT} >> ${TRAIN_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_PREFIX}.spm.${TGT_LANG_SHORT} >> ${TRAIN_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_PREFIX_REV}.spm.${SRC_LANG_SHORT} >> ${TRAIN_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_PREFIX_REV}.spm.${TGT_LANG_SHORT} >> ${TRAIN_SRC}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_FINAL}.spm.${SRC_LANG_SHORT} >> ${DEV_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_FINAL}.spm.${TGT_LANG_SHORT} >> ${DEV_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_FINAL}.spm.${SRC_LANG_SHORT} >> ${DEV_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_FINAL}.spm.${TGT_LANG_SHORT} >> ${DEV_SRC}

done

:|paste -d ' ||| ' ${TRAIN_SRC} - - - - ${TRAIN_TGT} > "${data_dir}/train"
shuf "${data_dir}/train" > "${data_dir}/train.shuf"
awk -F ' \\|\\|\\| ' '{print $1}' "${data_dir}/train.shuf" > "${data_dir}/train.shuf.src"
awk -F ' \\|\\|\\| ' '{print $2}' "${data_dir}/train.shuf" > "${data_dir}/train.shuf.tgt"

:|paste -d ' ||| ' ${DEV_SRC} - - - - ${DEV_TGT} > "${data_dir}/dev"
shuf "${data_dir}/dev" > "${data_dir}/dev.shuf"
awk -F ' \\|\\|\\| ' '{print $1}' "${data_dir}/dev.shuf" > "${data_dir}/dev.shuf.src"
awk -F ' \\|\\|\\| ' '{print $2}' "${data_dir}/dev.shuf" > "${data_dir}/dev.shuf.tgt"

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/train.shuf \
    --validpref ${data_dir}/dev.shuf \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir} \
    --thresholdtgt 0 --thresholdsrc 0 --seed ${seed} \
    --amp --workers 32

for SRC_LANG_LONG in "${mono_langs[@]}"
do
    SRC_LANG="${SRC_LANG_LONG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG^^}"

    MONO_DIR="${datasets_dir}/mono-${SRC_LANG}"
    TRAIN_PREFIX="${base_dir}/${prefix}/mono-${SRC_LANG}/train"

    ${SPM}/spm_encode --model=${MODEL} < "${TRAIN_PREFIX}.mono${SRC_LANG}noised" > "${TRAIN_PREFIX}.spm.mono${SRC_LANG}noised"
    ${SPM}/spm_encode --model=${MODEL} < "${TRAIN_PREFIX}.mono${SRC_LANG}target" > "${TRAIN_PREFIX}.spm.mono${SRC_LANG}target"

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' "${TRAIN_PREFIX}.spm.mono${SRC_LANG}noised" >> "${MONO_SRC}"
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' "${TRAIN_PREFIX}.spm.mono${SRC_LANG}target" >> "${MONO_TGT}"

done

:|paste -d ' ||| ' "${data_dir}/mono/train.src" - - - - "${data_dir}/mono/train.tgt" > "${data_dir}/mono/train"
shuf "${data_dir}/mono/train" > "${data_dir}/mono/train.shuf"
awk -F ' \\|\\|\\| ' '{print $1}' "${data_dir}/mono/train.shuf" > "${data_dir}/mono/train.shuf.src"
awk -F ' \\|\\|\\| ' '{print $2}' "${data_dir}/mono/train.shuf" > "${data_dir}/mono/train.shuf.tgt"

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref "${data_dir}/mono/train.shuf" \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir}/mono \
    --thresholdtgt 0 --thresholdsrc 0 --seed ${seed}\
    --amp --workers 32

sbatch --export=rr=$rr,seed=$seed train.sh

