---
layout: page
permalink: /publications/
title: publications
description: "* indicates equal contribution. Please check my <a href='https://scholar.google.com/citations?user=LbDh_igAAAAJ&hl=en'>Google Scholar</a> for the most up-to-date list on my research!"
years: [2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
