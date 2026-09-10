API reference
=============

The supported surface is what each module's own ``__all__`` lists.  Anything
else is private and may change without a major version.

Evaluating a mass function needs :doc:`model` and :doc:`box` alone, which
between them import nothing beyond numpy and JAX.  :doc:`target`,
:doc:`generate` and :doc:`fit` are the offline half that built the weights.

.. toctree::
   :maxdepth: 1

   model
   box
   target
   generate
   fit
