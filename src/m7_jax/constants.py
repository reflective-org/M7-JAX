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

# --- Multi-species registry and 18-component table (#1017) -----------------
# Species order SO4,BC,OC,SS,DU,WAT fixed by mo_ham_species.F90's AEROSOL
# species blocks: SO4 (lines 289-309), BC (lines 338-352), OC in the HAM_M7
# branch (lines 356-372), SS (lines 380-395), DU (lines 399-413), WAT
# (lines 417-432). mo_ham_species.F90 stores density in kg m-3 (native SI);
# CDCONV below converts to the g cm-3 the M7 core computes with internally.
# Mirrored verbatim by tools/m7_oracle.py's DRIVER speclist assignment
# (m7_oracle.py:142-145), which this harness's reference executable reads.
SPECIES = ('SO4', 'BC', 'OC', 'SS', 'DU', 'WAT')
SPECIES_MOLAR_MASS = (SO4_MOLAR_MASS, 12.010, 180., 58.443, 250., 18.0)  # g/mol
SPECIES_DENSITY = (1841., 2000., 2000., 2165., 2650., 1000.)  # kg/m3
SPECIES_KAPPA = (SULFATE_KAPPA, 0.0, 0.06, 1.0, 0.0, 0.0)
SPECIES_ELECTROLYTE = (True, False, False, True, False, False)

# The 18-entry `aerocomp` table (naerocomp=18): species and size-class (mode)
# each component belongs to, in HAM aerocomp order. Real HAM assembles this at
# runtime from the active species/mode configuration rather than as a literal
# table; the fixed numbering here is set by the #1017 lead's state layout
# (TASK.md) and mirrored by the oracle's `sid`/`mid` arrays
# (tools/m7_oracle.py:125-126), converted from Fortran's 1-based indices.
MODES = ('NS', 'KS', 'AS', 'CS', 'KI', 'AI', 'CI')  # 0..6; first 4 are soluble (nsol=4)
COMPONENT_SPECIES = (0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3, 4,4,4,4)  # index into SPECIES
COMPONENT_MODE =    (0,1,2,3, 1,2,3,4, 1,2,3,4, 2,3, 2,3,5,6)  # index into MODES
NSOL = 4  # number of soluble modes (mo_ham_m7ctl.F90 nsol), i.e. MODES[:NSOL]

# Unit/density conversion factors used by m7_averageproperties, m7_kappa and
# m7_dconc (mo_ham_m7.F90), kept as named constants instead of the inline
# literals the Fortran repeats under several local names (fac_ug/zfac2cgs are
# the same 1e-12 value under two names).
FAC_UG = 1e-12     # ug m-3 -> g cm-3 (mo_ham_m7.F90:825 fac_ug, :2953 zfac2cgs)
FAC_MOLEC = 1.0    # sulphate mass/number need no unit change here (mo_ham_m7.F90:826 fac_molec)
CDCONV = 1e-3      # density kg m-3 -> g cm-3 (mo_ham_m7ctl.F90:202)
CEPSSEC = 1e-25    # divide-by-zero guard in m7_kappa's volume-weighted kappa (mo_ham_m7.F90:986)
ZFACSO4M = H2SO4_MOLAR_MASS/AVOGADRO     # sulphate molec cm-3 -> g cm-3 (mo_ham_m7.F90:2951)
ZFACSO4V = ZFACSO4M/SULFATE_DENSITY      # sulphate molec cm-3 -> cm3 cm-3 (mo_ham_m7.F90:2952)

# Host (mo_ham_subm.f90::ham_subm_interface) tracer <-> native unit factor.
# mo_ham_subm.f90 is a BSD-3 HAM module that does not ship in this checkout;
# this literal (and the 1e9/1e-6 factors used directly in interface.py) is
# specified by the #1017 lead (TASK.md), not read from a Fortran file here.
# It is deliberately NOT jcm's full-precision AVOGADRO (6.02214179e23): the
# host interface uses this truncated constant, and the M7 core above uses
# AVOGADRO; porting both exactly, inconsistency included, is the faithful
# choice (see docs/model.md's other documented native unit distinctions).
HAM_SUBM_AVOGADRO = 6.022e20
