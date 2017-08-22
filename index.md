---
layout: index
title: "About"
group : navigation
description: "TVM"
---
{% include JB/setup %}


TVM: Tensor IR Stack for Deep Learning Systems
==============================================
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Build Status](http://mode-gpu.cs.washington.edu:8080/buildStatus/icon?job=dmlc/tvm/master)](http://mode-gpu.cs.washington.edu:8080/job/dmlc/job/tvm/job/master/)


TVM is a tensor intermediate representation(IR) stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

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
