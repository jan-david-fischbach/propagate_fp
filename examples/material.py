import numpy as np
def eps_ag(wfreq):
    # Also for silver we have parameters from fitting.
    # These are from fp^2 / [f0^2 - f^2 - i f g] with normal frequencies in THz.
    Agw0 = 2*np.pi*134.39519365 / 300
    Aggamma = 2*np.pi*10.61865288 / 300
    Agwp = 2*np.pi*1867.27147966 / 300

    return 1 + Agwp**2 / (Agw0**2 - wfreq**2 - 1j*Aggamma*wfreq)


def eps_cav(wfreq, npoles=3):
    # Material data
    # See verify_data.py for explanation on how to transform these to the usual quantities.
    f0 = np.array([448.79110491874115, 438.2930673770547, 412.93727009075883])
    intensity = np.array([1.04500718, 1.63227866, 14.84804589])
    damping = np.array([6.2, 6.0, 5.3])

    f0 = f0[-npoles:]
    intensity = intensity[-npoles:]
    damping = damping[-npoles:]

    gamma = damping*2*np.pi
    w0 = 2*np.pi*f0
    wp = np.sqrt(intensity*gamma*w0)
    # These frequencies are now in THz scale. For numerical sanity, we're going to be unitless;
    #   freq = 1 is 300 THz
    #   length = 1 is 1 micrometer
    # Speed of light is now 3e8 m/s which is good enough.
    w0 /= 300
    wp /= 300
    gamma /= 300

    eps_background = 1.6

    eps = eps_background*np.ones(wfreq.shape, dtype='complex')
    for i in range(npoles):
        eps += wp[i]**2 / (w0[i]**2 - wfreq**2 - 1j*wfreq*gamma[i])

    return eps