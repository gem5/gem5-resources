---
layout: default
title: "gem5 resources"
permalink: "/"
---

## gem5 Resources

gem5 resources is a repository providing sources for artifacts known and proven compatible with the [gem5 architecture simulator](https://gem5.org).
These resources are not necessary for the compilation or running of gem5, but may aid users in producing certain simulations.

## Why gem5 Resources?

gem5 has been designed with flexibility in mind. Users may simulate a wide variety of hardware, with an equally wide variety of workloads.
However, requiring users to find and configure workloads for gem5 (their own disk images, their own OS boots, their own tests, etc.) is a significant investment, and a hurdle to many.

The purpose of gem5 resources is therefore *to provide a stable set of commonly used resources, with proven and documented compatibility with gem5*.
In addition to this, gem5 resources also puts emphasis on *reproducibility of experiments* by providing citable, stable resources, tied to a particular release of gem5.

## Using gem5 resources

If you find one of the gem5-resources useful be sure to cite both the resources (see the README) and the [gem5art and gem5-resources paper](https://arch.cs.ucdavis.edu/assets/papers/ispass21-gem5art.pdf).

```bibtex
@inproceedings{bruce2021gem5art,
  title={Enabling Reproducible and Agile Full-System Simulation},
  author={Bruce, Bobby R. and Akram, Ayaz and Nguyen, Hoa and Roarty, Kyle and Samani, Mahyar and Fariborz, Marjan and Trivikram, Reddy and Sinclair, Matthew D. and Lowe-Power, Jason},
  booktitle={In Proceedings of the 2021 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS '21)},
  year={2021},
  organization={IEEE}
}

```

## List of current resources

<ul>
{% for page in site.pages %}
{% if page.path contains 'src' %}
{% include resource-brief.html page=page %}
{% endif %}
{% endfor %}
</ul>

## UNDER CONSTRUCTION

This website is under construction.
You can see the following links for the current information on gem5-resources.
More information will be here soon!

* [Documentation on gem5-resources](http://www.gem5.org/documentation/general_docs/gem5_resources/)
* [Source for gem5-resources](https://gem5.googlesource.com/public/gem5-resources/+/refs/heads/stable/)
* [README for gem5-resources]({{'README' | relative_url}})
* [The gem5art and gem5-resources paper](https://arch.cs.ucdavis.edu/assets/papers/ispass21-gem5art.pdf)
