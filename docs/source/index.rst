Pychastic Documentation
=======================

.. toctree::
   :hidden:
   :maxdepth: 1
   
   Tutorial (molecular dynamics) <moleculardynamics>
   Tutorial (derivatives pricing) <optionpricing>
   Advanced features <advancedfeatures>
   Minimal examples <examples>
   Detailed reference <reference>

Pychastic is a user friendly oriented stochastic differential equation
integrator package for both scalar and vector stochastic differential
equations.

Stochastic differential equations find use in all places where "noise" is 
important part of the problem such as microscopic phenomena or pandemic
dynamics. The framework of Ito calculus gained somewhat mainstream 
attention since the model of Black-Sholes allowed for accurate pricing of
options.

How to install
''''''''''''''

Easiest way to get the package from PyPi is using pip. Simply run:

.. prompt:: bash $ auto

  $ python3 -m pip install pychastic

and you'll be good to go.


How to cite
'''''''''''
| *Pychastic: Precise Brownian dynamics using Taylor-Itō integrators in Python*
| Radost Waszkiewicz, Maciej Bartczak, Kamil Kolasa, Maciej Lisicki
| SciPost Phys. Codebases 11 **(2023)**
| `doi.org/10.21468/SciPostPhysCodeb.11 <https://scipost.org/10.21468/SciPostPhysCodeb.11>`_.


Next steps
''''''''''

* :doc:`Tutorial - molecular dynamics themed<moleculardynamics>`.
* :doc:`Tutorial - derivatives pricing themed<optionpricing>`.
* :doc:`Advanced features <advancedfeatures>`.
* :doc:`Examples <examples>`.

Search tools
''''''''''''

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
