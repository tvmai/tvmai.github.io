---
layout: index
title: "About"
group : navigation
description: "TVM"
---
{% include JB/setup %}


TVM: End to End Optimization stack for AI Frameworks
====================================================
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=dmlc/tvm/master)](http://mode-gpu.cs.washington.edu:8080/job/dmlc/job/tvm/job/master/)


TVM stack is a unified optimization stack that will close the gap between the productivity-focused deep learning frameworks,
and the performance- or efficiency-oriented hardware backends. The project contains the following components:
- [TVM](https://github.com/dmlc/tvm) Tensor IR Stack for Deep Learning Systems
- [NNVM](https://github.com/dmlc/nnvm) Graph IR stack for Deep Learning Systems
- NNVM compiler: open compiler for AI Frameworks (shares same repo with NNVM)

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
