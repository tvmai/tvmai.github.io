---
layout: index
title: "TVM Stack"
order : 0
description: "TVM"
---
{% include JB/setup %}


Open End to End AI Compiler Stack
====================================================
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=dmlc/tvm/master)](http://mode-gpu.cs.washington.edu:8080/job/dmlc/job/tvm/job/master/)


TVM is an open deep learning compiler stack for cpu, gpu and specialized accelerators.
It aims to close the gap between the productivity-focused deep learning frameworks,
and the performance- or efficiency-oriented hardware backends.

Checkout our techreport [TVM: End-to-End Optimization Stack for Deep Learning](https://arxiv.org/abs/1802.04799)

{:center: style="text-align: center"}
![image](/images/main/stack_tvmlang.png){: width="90%"}
{:center}

News
----
<ul>
{% for post in site.posts %}
<li> <span>{{ post.date | date: "%b %-d, %Y" }} :
   <a class="post-link" href="{{ post.url | prepend: site.baseurl }}.html">{{ post.title }}</a>
</span>
</li>
{% endfor %}
</ul>
