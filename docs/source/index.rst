
GUNDAM
--------

**GUNDAM** is a data manager that utilizes language models to efficiently handle textual data, which is built upon `PyTorch <https://pytorch.org>`_.
GUNDAM is

- **Comprehensive**: GUNDAM provides data manager including our proposed miner, a GPT-2 based generator, and a demonstration retriever, and all of these components are extendable.
- **Flexible**: GUNDAM now supports GPT-2 language models (and we will extend it to more language models in future), with different sizes.
- **Efficient**: GUNDAM provides an efficient one-to-one miner (and we will extend it to one-to-poir and pair-to-pair miners in future) to check data quality..


.. toctree::
   :caption: Documentation:
   :maxdepth: 2

   manager/manager.rst
   miner/miner.rst
   generator/generator.rst
   retriever/retriever.rst

.. toctree::
   :caption: Examples
   :maxdepth: 2

   examples/gpt2med.rst

.. toctree::
   :caption: API Documentation
   :maxdepth: 2

   api/api.rst


Citing
------

If you find GUNDAM useful, please cite it in your publications.

.. code-block:: bibtex

      @software{GUNDAM,
        author = {Jiarui Jin, Yuwei Wu, Mengyue Yang, Xiaoting He, Weinan Zhang, Yiming Yang, Yong Yu, and Jun Wang},
        title = {GUNDAM: A Data-Centric Manager for Your Plug-in Data with Language Models},
        year = {2023},
        publisher = {GitHub},
        journal = {GitHub repository},
        version = {0.0},
        howpublished = {\url{https://github.com/GUNDAM-Labet/GUNDAM}},
      }


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
