Tutorial 2: compare the extended theis solution
===============================================

We provide an extended theis solution, that incorporates the effectes of a
heterogeneous transmissivity field on a pumping test.

In the following this extended solution is compared to the standard theis
solution for well flow. You can nicely see, that the extended solution represents
a transition between the theis solutions for the geometric- and harmonic-mean
transmissivity.

.. code-block:: python

    import numpy as np
    from matplotlib import pyplot as plt
    from anaflow import theis


    time = [10, 100, 1000]
    rad = np.geomspace(0.1, 10)

    head = theis(rad=rad, time=time, T=1e-4, S=1e-4, Qw=-1e-4)

    for i, step in enumerate(time):
        plt.plot(rad, head[i], label="Theis(t={})".format(step))

    plt.legend()
    plt.show()


.. image:: pics/02_call_ext_theis.png
   :width: 400px
   :align: center
