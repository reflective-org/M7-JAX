"""Pinned native M7 constants; keep legacy values and CGS units for parity."""
import math
import numpy as np

PI = 3.14159265358979323846
AVOGADRO = 6.02214179e23
BK = 1.38e-16  # erg/K, mo_ham_m7ctl
RERG = 8.314e7  # erg/(mol K)
H2SO4_MOLAR_MASS = 98.0734  # g/mol, microphysics
SO4_MOLAR_MASS = 96.0631  # g/mol, species registry used by kappa hydration
SULFATE_DENSITY = 1.841  # g/cm3
SULFATE_KAPPA = .60
CMIN_AERNL = 1e-10  # cm-3
CMIN_AERML = 1e-15
CMINVOL = 1e-23  # cm3
CMINRAD = 1e-8  # cm
CMINRHO = 1e-10  # g/cm3
ZMIN_COAG = float(np.float32(1e-15))  # native unsuffixed literal
ACCOMMODATION = (1., 1., 1., 1., .3, .3, .3)
SIGMA = (1.59, 1.59, 1.59, 2., 1.59, 1.59, 2.)
SIGMA_LOG = tuple(math.log(x) for x in SIGMA)
CMR2RAM = tuple(math.exp(1.5*math.log(x)**2) for x in SIGMA)
RAM2CMR = tuple(1/x for x in CMR2RAM)
MODE_RADIUS_BOUNDS = (.0005e-4, .005e-4, .05e-4, .5e-4)  # cm
CRITICAL_MOLECULES = MODE_RADIUS_BOUNDS[0]**3*PI*AVOGADRO*SULFATE_DENSITY/H2SO4_MOLAR_MASS/.75
COAG_MODE_PAIRS = ((0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),
                   (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(2,2),(2,4),(4,4))
WVB = (95.80188,-28.5257,-1.082153,.1466501,-20627.51,.0461242,-.003935,
       -3.36115,-.00024137,.067938345,.00000649899,8616124.373,1.168155578,
       -.021317481,.000270358,-1353332314.,-.002403805)
GMB = (1.036391467,.00728531,-.011013887,-.068887407,.001047842,
       .001049607,.000740534,-1.081202685,-.0000029113)
