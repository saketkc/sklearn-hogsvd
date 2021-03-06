.. -*- mode: rst -*-

|Travis|_  |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/saketkc/sklearn-hogsvd.svg?branch=master
.. _Travis: https://travis-ci.org/saketkc/sklearn-hogsvd

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-hogsvd/badge/?version=latest
.. _ReadTheDocs: https://sklearn-hogsvd.readthedocs.io/en/latest/?badge=latest

sklhogsvd - Higher order generalized singular value decomposition 
=================================================================

.. _scikit-learn: https://scikit-learn.org

**sklhogsvd** is a ``scikit-learn`` compatible python package to perform
higher order generalized singular value decomposition (HOGSVD) as described
in [Ponnapalli2011]_ with an additional support for ensuring
orthogonality in the "arraylet" space.


- The orthogonalization trick is summarized in the PDF `here <https://saket-choudhary.me/pdfs/HOGSVD_orthogonalization.pdf>`_.

- A poster demonstrating its utility in finding orthologous genes (based on expression) is `here <https://f1000research.com/posters/7-1853>`_. 

- A work in progress writeup is `here <https://saket-choudhary.me/pdfs/Higher_Order_Generalized_SVD_based_alignment_free_method_for_inferring_orthologous_genes_across_species.pdf>`_.

`Demo notebook  <./notebooks/demo.ipynb>`_.


.. [Ponnapalli2011] Ponnapalli, S.P., Saunders, M.A., Van Loan, C.F., and Alter, O. (2011). A Higher-Order Generalized Singular Value Decomposition for Comparison of Global mRNA Expression from Multiple Organisms. PLoS ONE 6, e28072. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0028072


.. _documentation: https://sklearn-hogsvd.readthedocs.io/en/latest/quick_start.html

