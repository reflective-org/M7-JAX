"""Actual native M7 process/core oracle, restricted to pure sulfate and nsoa=0.

No numerical routine is translated. A PUBLIC visibility overlay exposes original
private leaves; metadata-only stubs initialize a documented 18-component registry.
The authentic kappa NetCDF table is serialized losslessly for the Fortran reader.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reference' / 'fortran'
if not SOURCE.exists():
    SOURCE = ROOT
# Source corrections applied at build time. The SALSA-box copy of
# m7_averageproperties reuses ``ztmp1`` for the insoluble-mode density and then
# takes the cube root of that DENSITY as if it were the mean particle volume,
# so every insoluble count-median radius comes out as ram2cmr*(rho/(4pi/3))**(1/3)
# (~0.57 cm for BC). ECHAM6.3-HAM2.3 r7492 (mo_ham_m7.f90:862-874, the reference
# configuration) keeps ``ztmp1`` as the volume. The reference file stays
# byte-identical to its pinned hash; the oracle replaces exactly this text and
# records it in the provenance.
SOURCE_CORRECTIONS = (
    ('mo_ham_m7.F90',
     "        ztmp1(1:kproma,:) = zinsmas(1:kproma,:,jclass) / ztmp1(1:kproma,:)\n"
     "        prhop(1:kproma,:,jclass) = MERGE(ztmp1(1:kproma,:), prhop(1:kproma,:,jclass), ll1(1:kproma,:))\n",
     "        prhop(1:kproma,:,jclass) = MERGE(zinsmas(1:kproma,:,jclass) / ztmp1(1:kproma,:), prhop(1:kproma,:,jclass), ll1(1:kproma,:))\n",
     'insoluble-mode radius from mean particle volume, as ECHAM6.3-HAM2.3 r7492 mo_ham_m7.f90:862-874'),
)


def _corrected(name, text):
    for target, old, new, _ in SOURCE_CORRECTIONS:
        if target == name:
            if text.count(old) != 1:
                raise ValueError(f'Source correction anchor changed in {name}')
            text = text.replace(old, new, 1)
    return text


SOURCES = ('parkind1.F90', 'mo_kind.F90', 'mo_math_constants.F90', 'mo_physical_constants.F90', 'mo_ham_m7ctl.F90', 'mo_ham_m7_nucl.F90', 'mo_ham_kappa.F90', 'mo_ham_m7.F90')
OPERATIONS = ('coaset', 'dnum', 'condensation_sink', 'produce_and_condense', 'nucleate', 'vehkamaeki', 'properties', 'hydrate', 'repartition', 'full', 'cumulative_normal', 'kazil_lovejoy')

def _routine(source, name):
    match = re.search(r'^\s*SUBROUTINE\s+' + name + r'\b.*?^\s*END SUBROUTINE\s+' + name + r'\b[^\n]*', source, re.M | re.S | re.I)
    if not match:
        raise ValueError(name)
    return match.group(0)

STUBS = '''module mo_species
use mo_kind, only: dp
implicit none
integer, parameter :: nmaxspec=6
type species_type
real(dp) :: moleweight=0, density=0, kappa=0
logical :: lelectrolyte=.false.
integer :: iaerocomp(7)=0
end type
type(species_type), target :: speclist(6)
end module
module mo_ham
use mo_kind, only: dp
use mo_species
implicit none
integer, parameter :: HAM_M7=2,naeroclass(3)=[1,7,17],nclass=7,nsol=4,naerocomp=18
integer, parameter :: nsoa=0,nsoaspec=0,subm_ngasspec=2,subm_naerospec=6
integer, parameter :: immr2ug=1,immr2molec=2
integer :: subm_aerospec(6)=[1,2,3,4,5,6],subm_aero_idx(6)=[1,2,3,4,5,6]
integer :: subm_aerounitconv(6)=[2,1,1,1,1,1]
real(dp), parameter :: sigma_fine=1.59_dp,sigma_coarse=2._dp,mw_so4=96.0631_dp
logical :: lscoag=.true.,lscond=.true.
type component_type
integer :: spid=0,iclass=0
type(species_type),pointer :: species=>null()
end type
type class_type
character(40) :: classname=''
character(2) :: shortname=''
integer :: self=0
logical :: lsoluble=.false.,lsed=.false.,lsoainclass=.false.,lactivation=.false.
end type
type(component_type) :: aerocomp(18)
type(class_type) :: sizeclass(7)
end module
module mo_ham_species
integer,parameter :: id_so4=1,id_bc=2,id_oc=3,id_ss=4,id_du=5,id_wat=6
end module
module mo_ham_subm_species
integer,parameter :: isubm_so4g=2,isubm_wat=6
end module
module mo_time_control
use mo_kind,only:dp
real(dp) :: time_step_len=1._dp
end module
module mo_ham_wetdep_data
use mo_kind,only:dp
real(dp),allocatable :: csr_strat_wat(:),csr_strat_mix(:),csr_strat_ice(:),csr_conv(:),cbcr(:),cbcs(:)
end module
module mo_ham_soa
type soa_type
integer :: spid_soa=0
logical :: lvolatile=.false.
end type
type(soa_type) :: soaprop(1)
end module
'''

# Additional metadata-only stub modules pulled in by mo_ham_m7.F90's
# '#ifdef HAMMOZ' blocks (m7, m7_prod_cond, m7_nuck, m7_dnum, m7_delcoa):
# the HAM diagnostic streams, the G3B forest fraction, delta_time, the
# HAMMOZ timer wrappers and the nucleation-statistics sampler. None of this
# is science -- every routine here is either a no-op or a visibility-only
# accumulator never read back by the operations this harness exercises.
# Used only when build_oracle(..., hammoz=True); the default build (and
# every existing test) is untouched.
HAMMOZ_STUBS = '''module mo_control
logical :: ltimer = .false.
end module
module mo_hammoz_timer
integer, parameter :: timer_ham_m7_avg=1, timer_ham_m7_hygro=2, timer_ham_m7_cs=3, &
  timer_ham_m7_cond=4, timer_ham_m7_nucl=5, timer_ham_m7_dnum=6,                   &
  timer_ham_m7_redistr=7, timer_ham_m7_coaset=8, timer_ham_m7_delcoa=9,            &
  timer_ham_m7_concoag=10
contains
subroutine timer_start(id)
integer :: id
end subroutine
subroutine timer_stop(id)
integer :: id
end subroutine
end module
module mo_ham_m7_nucl_diag
use mo_kind, only: dp
contains
subroutine sample_nucl_statistics(kproma,kbdim,klev,ptp1,prhp1,pso4g,pipr,pcs,paclc,pgrvol)
integer :: kproma,kbdim,klev
real(dp) :: ptp1(kbdim,klev),prhp1(kbdim,klev),pso4g(kbdim,klev),pipr(kbdim,klev)
real(dp) :: pcs(kbdim,klev),paclc(kbdim,klev),pgrvol(kbdim,klev)
end subroutine
end module
module mo_ham_streams
use mo_kind, only: dp
real(dp), allocatable :: d_nuc_so4(:,:), d_cond_so4(:,:)
end module
module mo_memory_g3b
use mo_kind, only: dp
real(dp), allocatable :: forest(:,:)
end module
'''

# mo_time_control.F90's stub in STUBS declares only time_step_len; m7_nuck
# and m7_prod_cond additionally need delta_time under '#ifdef HAMMOZ'. This
# exact-anchor replace (mirrors _corrected's pattern) keeps the hammoz=False
# metadata.F90 text byte-identical to before.
_TIME_CONTROL_STUB = 'module mo_time_control\nuse mo_kind,only:dp\nreal(dp) :: time_step_len=1._dp\nend module\n'
_TIME_CONTROL_STUB_HAMMOZ = ('module mo_time_control\nuse mo_kind,only:dp\n'
    'real(dp) :: time_step_len=1._dp\nreal(dp) :: delta_time=1._dp\nend module\n')


def _hammoz_metadata(stubs):
    if stubs.count(_TIME_CONTROL_STUB) != 1:
        raise ValueError('mo_time_control stub anchor changed')
    return stubs.replace(_TIME_CONTROL_STUB, _TIME_CONTROL_STUB_HAMMOZ, 1) + HAMMOZ_STUBS


DRIVER = '''program oracle
use mo_kind, only: dp
use mo_physical_constants, only: avo
use mo_ham
use mo_ham_m7ctl
use mo_ham_m7
use mo_ham_m7_nucl
use mo_ham_kappa, only: start_kappa
use mo_time_control
implicit none
integer :: nc,reps,i,j,k,rep,op,coag,cond,water,scheme,steps,step
integer :: sid(18)=[1,1,1,1,2,2,2,2,3,3,3,3,4,4,5,5,5,5]
integer :: mid(18)=[1,2,3,4,2,3,4,5,2,3,4,5,3,4,3,4,6,7]
integer :: switches(4)
real(dp) :: number(1,1,7),mass(1,1,18),radius(1,1,7),density(1,1,7),dry(1,1,4),ww(1,1,7)
real(dp) :: ttn(1,1,18),delta(1,1,18),temp(1,1),pres(1,1),rh(1,1),gas(1,1),prod(1,1)
real(dp) :: cs(1,1),csi(1,1,7),cloud(1,1),newn(1,1),s5(1,1),s6(1,1),s7(1,1)
real(dp) :: coeff(1,1,16),r0(1,1,7),d0(1,1,7),pgas(1,1,2),ion(1,1),dz(1,1),grvol(1,1),pbl(1),diag(1,1,50)
real(dp) :: rate(1,1),cluster(1,1),elapsed,begin,finish
real(dp),allocatable :: input(:,:), output(:,:)
character(32) :: arg
call get_command_argument(1,arg)
read(arg,*) op
call m7_initialize
iso4ns=1;iso4ks=2;iso4as=3;iso4cs=4
ibcks=5;ibcas=6;ibccs=7;ibcki=8
iocks=9;iocas=10;ioccs=11;iocki=12
issas=13;isscs=14;iduas=15;iducs=16;iduai=17;iduci=18
speclist%moleweight=[96.0631_dp,12.01_dp,180._dp,58.443_dp,250._dp,18._dp]
speclist%density=[1841._dp,2000._dp,2000._dp,2165._dp,2650._dp,1000._dp]
speclist%kappa=[.6_dp,0._dp,.06_dp,1._dp,0._dp,0._dp]
speclist(1)%lelectrolyte=.true.;speclist(4)%lelectrolyte=.true.
do j=1,18
 aerocomp(j)%spid=sid(j);aerocomp(j)%iclass=mid(j)
 aerocomp(j)%species=>speclist(sid(j))
 speclist(sid(j))%iaerocomp(mid(j))=j
enddo
call start_kappa
open(unit=91,file='constants.txt',status='replace')
write(91,'(*(ES26.17E3,1X))') avo,bk,rerg,r_kcal,mw_so4,wh2so4,dh2so4,dh2o,critn,cmin_aernl,cmin_aerml,cminvol,cminrad,cminrho,cdconv,crh,so4_coating_threshold
write(91,'(*(ES26.17E3,1X))') sigma,sigmaln,cmr2ram,ram2cmr,cmr2ras,cmr2mmr,cmedr2mmedr,caccso4,crdiv
close(91)
read(*,*) nc,reps,steps
allocate(input(88,nc),output(107,nc))
do i=1,nc
 read(*,*) input(:,i)
enddo
elapsed=0._dp
call cpu_time(begin)
do rep=1,reps
 do i=1,nc
 ! Input row layout (1-based), kept in sync with INPUT_LAYOUT in this module:
 ! number 1:7, radius 8:14, density 15:21, mass 22:39 (HAM aerocomp order),
 ! delta 40:57, temp 58, pres 59, dt 60, rh 61, gas 62, prod 63, cloud 64,
 ! cs 65, csi 66:72, newn 73, coag 74, cond 75, water 76, nsnucl 77,
 ! dry 78:81, s5 82, s6 83, s7 84, nonucl 85, ion 86, forest 87, pbl 88.
 number(1,1,:)=input(1:7,i);radius(1,1,:)=input(8:14,i);density(1,1,:)=input(15:21,i)
 mass(1,1,:)=input(22:39,i)
 delta(1,1,:)=input(40:57,i)
 temp=input(58,i);pres=input(59,i);time_step_len=input(60,i);rh=input(61,i)
 gas=input(62,i);prod=input(63,i);cloud=input(64,i);cs=input(65,i)
 csi(1,1,:)=input(66:72,i);newn=input(73,i)
 lscoag=input(74,i)==1;lscond=input(75,i)==1;nwater=int(input(76,i));nsnucl=int(input(77,i))
 dry(1,1,:)=input(78:81,i);s5=input(82,i);s6=input(83,i);s7=input(84,i)
 if (nsnucl<0.or.nsnucl>1) error stop 'unsupported nucleation scheme'
 ! nonucl/forest take effect only in a HAMMOZ build (organic nucleation is
 ! compiled out of the standalone source otherwise); ion and pbl are m7 args.
 nonucl=int(input(85,i));ttn=0;ww=0;coeff=0;rate=0;cluster=0;pgas=0
 ion=input(86,i);dz=1;grvol=1;pbl=input(88,i);diag=0
 do step=1,steps
 select case(op)
 case(1)
 call m7_coaset(1,1,1,1,number,temp,pres,radius,density,coeff)
 case(2)
 r0=radius;d0=density
 call m7_averageproperties(1,1,1,1,number,mass,ttn,r0,d0)
 call m7_dnum(1,1,1,1,mass,number,temp,pres,radius,density,s5,s6,s7,newn,delta,ttn)
 case(3)
 call m7_h2so4_cs(1,1,1,temp,pres,number,radius,cs,csi)
 case(4)
 call m7_prod_cond(1,1,1,1,gas,prod,mass,s5,s6,s7,cs,csi,cloud,dz)
 case(5)
 call m7_nuck(1,1,1,1,pres,temp,rh,cloud,ion,gas,cs,newn,delta,dz,pbl)
 case(6)
 call nucl_vehkamaeki(1,1,1,temp,rh,gas,rate,cluster)
 case(7)
 call m7_averageproperties(1,1,1,1,number,mass,ttn,radius,density)
 case(8)
 call m7_averageproperties(1,1,1,1,number,mass,ttn,radius,density)
 if(nwater==1) then
 call m7_kappa(1,1,1,1,rh,number,ttn,temp,dry,radius,density,ww)
 else
 call m7_equiz(1,1,1,1,pres,ttn,temp,rh,radius,dry,density,ww,number)
 call m7_equimix(1,1,1,1,pres,ttn,temp,rh,radius,dry,density,ww,number)
 call m7_equil(1,1,1,1,rh,mass,number,radius,dry,ww,density,temp)
 endif
 case(9)
 call m7_dconc(1,1,1,1,mass,number,dry)
 case(10)
 call m7(1,1,1,1,pres,temp,rh,pgas,gas,prod,mass,number,radius,dry,density,ww,ion,cloud,dz,grvol,pbl,diag)
 case(11)
 call m7_cumulative_normal(gas(1,1),rate(1,1),cluster(1,1))
 case(12)
 ! 'relative_humidity' in the shared input layout is the 0-1 fraction
 ! (same convention as nucl_vehkamaeki/m7_nuck's own prhp1); m7_nuck
 ! converts to percent before its nucl_kazil_lovejoy call (zrh=100*prhp1,
 ! mo_ham_m7.F90:2381), so this direct-call case does the same conversion.
 call nucl_kazil_lovejoy(1,1,1,temp,100._dp*rh,gas,cs,ion,rate,cluster)
 end select
 enddo
 output(:,i)=0
 output(1:7,i)=number(1,1,:);output(8:11,i)=mass(1,1,1:4);output(12:15,i)=delta(1,1,1:4)
 output(16:22,i)=radius(1,1,:);output(23:29,i)=density(1,1,:);output(30:33,i)=dry(1,1,:)
 output(34:40,i)=ww(1,1,:);output(41,i)=gas(1,1);output(42,i)=newn(1,1)
 output(43,i)=cs(1,1);output(44:50,i)=csi(1,1,:);output(51:66,i)=coeff(1,1,:)
 output(67,i)=rate(1,1);output(68,i)=cluster(1,1)
 output(69:86,i)=ttn(1,1,:);output(87:89,i)=[s5(1,1),s6(1,1),s7(1,1)]
 output(90:107,i)=mass(1,1,:)
 enddo
enddo
call cpu_time(finish)
write(*,'(ES26.17E3)') finish-begin
do i=1,nc
write(*,'(*(ES26.17E3,1X))') output(:,i)
enddo
end program
'''

# Exact-anchor edits applied only for the hammoz=True driver (see
# _hammoz_metadata): import the G3B forest fraction and HAM streams, size
# them once for the harness's single (kproma=1, krow=1) column, and feed
# forest(1,1) from input column 87 every case -- the one piece of wiring
# the task calls out explicitly. hammoz=False keeps driver.F90 byte-identical.
_USE_ANCHOR = 'use mo_time_control\nimplicit none'
_USE_HAMMOZ = ('use mo_time_control\nuse mo_memory_g3b, only: forest\n'
    'use mo_ham_streams, only: d_nuc_so4, d_cond_so4\nimplicit none')
_INIT_ANCHOR = 'call m7_initialize\n'
_INIT_HAMMOZ = ('call m7_initialize\n'
    'allocate(forest(1,1),d_nuc_so4(1,1),d_cond_so4(1,1))\n'
    'forest=0._dp;d_nuc_so4=0._dp;d_cond_so4=0._dp\n')
_PBL_ANCHOR = ' ion=input(86,i);dz=1;grvol=1;pbl=input(88,i);diag=0\n'
_PBL_HAMMOZ = ' ion=input(86,i);dz=1;grvol=1;pbl=input(88,i);diag=0;forest(1,1)=input(87,i)\n'


def _hammoz_driver(driver):
    for anchor, replacement in ((_USE_ANCHOR, _USE_HAMMOZ), (_INIT_ANCHOR, _INIT_HAMMOZ),
                                 (_PBL_ANCHOR, _PBL_HAMMOZ)):
        if driver.count(anchor) != 1:
            raise ValueError('driver.F90 hammoz anchor changed')
        driver = driver.replace(anchor, replacement, 1)
    return driver


# Kazil/Lovejoy driver variant (jax-gcm#1017 Kazil/GCR task, Part A): loads
# the real PARNUC table (start_nucl, below) once at startup and relaxes the
# nsnucl guard to admit 2. Independent of the hammoz=True anchors above, so
# both can apply together (the kazil-table build and the hammoz build edit
# disjoint text).
_NSNUCL_GUARD = "if (nsnucl<0.or.nsnucl>1) error stop 'unsupported nucleation scheme'\n"
_NSNUCL_GUARD_KAZIL = "if (nsnucl<0.or.nsnucl>2) error stop 'unsupported nucleation scheme'\n"


def _kazil_driver(driver):
    for anchor, replacement in ((_INIT_ANCHOR, 'call m7_initialize\ncall start_nucl\n'),
                                 (_NSNUCL_GUARD, _NSNUCL_GUARD_KAZIL)):
        if driver.count(anchor) != 1:
            raise ValueError('driver.F90 kazil anchor changed')
        driver = driver.replace(anchor, replacement, 1)
    return driver


def build_oracle(directory, checked=True, hammoz=False, kazil_table_path=None):
    """Build the native M7 oracle.

    ``hammoz=True`` compiles with ``-DHAMMOZ`` so m7_nuck's organic-nucleation
    branch (nonucl 1/2) is live, matching mo_ham_m7.F90:2410-2422. Everything
    this activates in turn (m7's/m7_dnum's/m7_delcoa's own '#ifdef HAMMOZ'
    timer and diagnostic blocks, pulled in because the whole file is compiled
    as one unit) gets metadata-only stubs -- see HAMMOZ_STUBS and
    _hammoz_driver. No science line changes; hammoz=False is untouched.

    ``kazil_table_path`` (jax-gcm#1017 Kazil/GCR task, Part A): the real
    PARNUC NetCDF table (``parnuc.15H2SO4.A0.total.nc``/``parnuc.15H2SO4.nc``,
    not shipped -- MPI-M/HAMMOZ data, from ``HAM_INPUT_DIR``). When given,
    ``nucl_kazil_lovejoy`` (the actual interpolation routine, lines 320-668)
    is compiled VERBATIM alongside its real module-level table arrays and a
    bespoke ``start_nucl`` (own code) that reads a binary serialization of
    the table -- the same pattern ``start_kappa`` below already uses for the
    kappa table, avoiding a real NetCDF Fortran library dependency for a
    harness that never otherwise needs one. ``nsnucl`` up to 2 is then
    accepted; without it (``None``, the default) the build is byte-identical
    to before: ``nucl_kazil_lovejoy`` stays the ``error stop`` stub and
    ``nsnucl`` stays capped at 1. Combining this with ``hammoz=True`` is not
    supported (both edit the same driver init anchor); raises if asked.
    """
    if hammoz and kazil_table_path is not None:
        raise NotImplementedError(
            'build_oracle does not support hammoz=True with a kazil_table_path '
            '(both edit the driver init anchor); build two separate oracles instead'
        )
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('gfortran')
    if compiler is None:
        raise RuntimeError('gfortran is required')
    source_text = {name: (SOURCE / name).read_text() for name in SOURCES}
    files = []
    def put(name, text):
        (directory / name).write_text(text)
        files.append(name)
    for name in SOURCES[:4]:
        put(name, source_text[name])
    put('metadata.F90', _hammoz_metadata(STUBS) if hammoz else STUBS)
    ctl = source_text['mo_ham_m7ctl.F90']
    put('mo_ham_m7ctl.F90', ctl.split('CONTAINS',1)[0] + '\nCONTAINS\n' + _routine(ctl,'m7_initialize') + "\nsubroutine sethamM7\nerror stop 'not used'\nend subroutine\nend module\n")
    kp = source_text['mo_ham_kappa.F90']
    metadata = kp[kp.index('       Rd_min = Rd(1)'):kp.index('       log_rh_step_compress = LOG(rh_step_compress)') + len('       log_rh_step_compress = LOG(rh_step_compress)')]
    put('mo_ham_kappa.F90', kp.split('CONTAINS',1)[0] + '''\nCONTAINS
subroutine start_kappa
integer :: u
open(newunit=u,file='kappa.bin',access='stream',form='unformatted',status='old')
read(u) N_Rd,N_T,N_rh,N_kappa
allocate(Rd(N_Rd),T(N_T),rh(N_rh),kappa(N_kappa),gf(N_Rd,N_T,N_rh,N_kappa))
read(u) Rd,T,rh,kappa,gf
close(u)
lut_kappa_version=2
''' + metadata + '\nend subroutine\nsubroutine term_kappa\nend subroutine\nend module\n')
    nucl=source_text['mo_ham_m7_nucl.F90']
    kazil='''subroutine nucl_kazil_lovejoy(kproma,kbdim,klev,ptemp,prh,ph2so4,pcs,pipr,ppfr,pns)
integer :: kproma,kbdim,klev
real(dp) :: ptemp(kbdim,klev),prh(kbdim,klev),ph2so4(kbdim,klev),pcs(kbdim,klev),pipr(kbdim,klev),ppfr(kbdim,klev),pns(kbdim,klev)
error stop 'Kazil table unavailable; unsupported scheme'
end subroutine
'''
    if kazil_table_path is not None:
        # Verbatim module-level table declarations (lines ~49-57) and the
        # verbatim nucl_kazil_lovejoy routine (320-668), plus a bespoke
        # start_nucl (own code, mirrors start_kappa's pattern below) that
        # reads a stream serialization of the real table instead of
        # ham_nucl_initialize's NetCDF calls -- same reasoning as
        # start_kappa: no real NetCDF Fortran library dependency needed for
        # a harness that only ever reads one pre-validated table.
        decl_start = nucl.index('INTEGER, PRIVATE :: t_n')
        decl_end = nucl.index('REAL(dp), PRIVATE, ALLOCATABLE :: pfr_table(:,:,:,:,:)')
        decl_end += len('REAL(dp), PRIVATE, ALLOCATABLE :: pfr_table(:,:,:,:,:)')
        table_decls = nucl[decl_start:decl_end]
        start_nucl = '''subroutine start_nucl
integer :: u
open(newunit=u,file='pfr.bin',access='stream',form='unformatted',status='old')
read(u) t_n,rh_n,h2so4_n,ipr_n,cs_n
allocate(t_table(t_n),rh_table(rh_n),h2so4_table(h2so4_n),ipr_table(ipr_n),cs_table(cs_n))
allocate(pfr_table(cs_n,ipr_n,h2so4_n,rh_n,t_n))
read(u) t_table,rh_table,h2so4_table,ipr_table,cs_table,pfr_table
close(u)
end subroutine
'''
        put('mo_ham_m7_nucl.F90',
            'module mo_ham_m7_nucl\nuse mo_kind,only:dp\n' + table_decls + '\ncontains\n'
            + '\n'.join(_routine(nucl,name) for name in ('nucl_vehkamaeki','nucl_activation','nucl_kinetic'))
            + '\n' + _routine(nucl,'nucl_kazil_lovejoy') + '\n' + start_nucl + 'end module\n')
    else:
        put('mo_ham_m7_nucl.F90','module mo_ham_m7_nucl\nuse mo_kind,only:dp\ncontains\n' + '\n'.join(_routine(nucl,name) for name in ('nucl_vehkamaeki','nucl_activation','nucl_kinetic')) + '\n' + kazil + 'end module\n')
    original=_corrected('mo_ham_m7.F90', source_text['mo_ham_m7.F90'])
    public='\nPUBLIC :: m7_coaset,m7_dnum,m7_h2so4_cs,m7_prod_cond,m7_nuck,m7_averageproperties,m7_kappa,m7_equiz,m7_equimix,m7_equil,m7_dconc\n'
    put('mo_ham_m7.F90', original.replace('CONTAINS',public+'CONTAINS',1))
    (directory/'mo_ham_m7.original.F90').write_text(original)
    driver = _hammoz_driver(DRIVER) if hammoz else DRIVER
    if kazil_table_path is not None:
        driver = _kazil_driver(driver)
    put('driver.F90', driver)
    if kazil_table_path is not None:
        from m7_jax.nucleation import load_kazil_lovejoy_table
        kt = load_kazil_lovejoy_table(kazil_table_path)
        with (directory/'pfr.bin').open('wb') as stream:
            # 1-D axes: byte order is irrelevant to Fortran's sequential
            # stream read. pfr: the table's natural (t,rh,h2so4,ipr,cs)
            # C-order array has cs varying fastest in memory, identical to
            # Fortran's column-major pfr_table(cs_n,ipr_n,h2so4_n,rh_n,t_n)
            # (cs is ALSO its fastest-varying subscript) -- so a plain
            # C-order byte dump needs no transpose/reorder.
            stream.write(np.array([len(kt.temperature), len(kt.relative_humidity),
                len(kt.h2so4), len(kt.ion_pair_rate), len(kt.condensation_sink)],
                dtype='=i4').tobytes())
            for axis in (kt.temperature, kt.relative_humidity, kt.h2so4,
                         kt.ion_pair_rate, kt.condensation_sink):
                stream.write(np.asarray(axis, dtype='=f8').tobytes())
            stream.write(np.ascontiguousarray(np.asarray(kt.log_pfr, dtype='=f8')).tobytes())
    table = ROOT/'reference'/'input'/'lut_kappa.nc'
    if not table.exists():
        table=ROOT/'lut_kappa.nc'
    from scipy.io import netcdf_file
    with netcdf_file(table,'r',mmap=False) as nc:
        if nc.Version != b'2.0':
            raise ValueError('Only authentic version 2.0 kappa table is supported')
        arrays=[np.array(nc.variables[name].data,dtype='=f8') for name in ('DryRadius','Temperature','RelativeHumidity','kappa','GF')]
    with (directory/'kappa.bin').open('wb') as stream:
        stream.write(np.array(arrays[-1].shape,dtype='=i4').tobytes())
        for value in arrays:
            stream.write(value.ravel(order='F').tobytes())
    flags=['-cpp','-O2','-fno-fast-math','-fno-lto','-ffree-line-length-none']
    if hammoz:
        flags = ['-DHAMMOZ'] + flags
    if checked:
        flags += ['-fcheck=all','-ffpe-trap=invalid,zero,overflow','-fbacktrace']
    executable=directory/'m7_oracle'
    start=time.perf_counter()
    result=subprocess.run([compiler,*flags,*files,'-o',str(executable)],cwd=directory,capture_output=True,text=True,timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    provenance={'checked':checked,'hammoz':hammoz,'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0], 'flags':flags,'source_sha256':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in SOURCES},'table_sha256':hashlib.sha256(table.read_bytes()).hexdigest(),'compile_seconds':time.perf_counter()-start,'overlay':'PUBLIC visibility only for complete mo_ham_m7; original ctl initialization and nucleation leaves extracted verbatim','registry':{'sulfate_components':[1,2,3,4],'species':['SO4','BC','OC','SS','DU','WAT'],'nsoa':0,'components':18},'excluded':['Kazil lookup nucleation'] if hammoz else ['Kazil lookup nucleation','organic nucleation'],'source_corrections':[{'file':f,'reason':r} for f,_,_,r in SOURCE_CORRECTIONS],'hammoz_stubs':['mo_control (ltimer)','mo_hammoz_timer (timer_start/timer_stop + ids)','mo_ham_m7_nucl_diag (sample_nucl_statistics)','mo_ham_streams (d_nuc_so4, d_cond_so4)','mo_memory_g3b (forest)','mo_time_control (+delta_time)'] if hammoz else []}
    subprocess.run([str(executable),'1'],input='0 1 1\n',text=True,capture_output=True,cwd=directory,check=True,timeout=30)
    compiled=np.loadtxt(directory/'constants.txt',max_rows=1)
    names='avo bk rerg r_kcal mw_so4 wh2so4 dh2so4 dh2o critn cmin_aernl cmin_aerml cminvol cminrad cminrho cdconv crh so4_coating_threshold'.split()
    provenance['compiled_constants']=dict(zip(names,compiled.tolist()))
    arrays=np.loadtxt(directory/'constants.txt',skiprows=1)
    for i,name in enumerate(('sigma','sigmaln','cmr2ram','ram2cmr','cmr2ras','cmr2mmr','cmedr2mmedr','caccso4')):
        provenance['compiled_constants'][name]=arrays[7*i:7*i+7].tolist()
    provenance['compiled_constants']['crdiv']=arrays[56:].tolist()
    provenance['generated_source_sha256']={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in files}
    provenance['table_binary_sha256']=hashlib.sha256((directory/'kappa.bin').read_bytes()).hexdigest()
    provenance['metadata_source_sha256']={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('mo_ham.F90','mo_ham_species.F90','mo_ham_init.F90')}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return executable,provenance


def run_oracle(executable, cases, operation='full', repetitions=1, steps=1, kazil=False):
    """``kazil=True`` (jax-gcm#1017 Kazil/GCR task, Part A): the executable was
    built with ``build_oracle(..., kazil_table_path=...)``, so nsnucl=2 (and
    the 'kazil_lovejoy' operation, which needs it) is valid; otherwise nsnucl
    is still capped at 0/1, matching the stub's ``error stop``.
    """
    if operation not in OPERATIONS:
        raise ValueError(operation)
    if operation == 'kazil_lovejoy' and not kazil:
        raise ValueError("operation='kazil_lovejoy' requires an oracle built with kazil_table_path (kazil=True)")
    candidates=[np.asarray(v).shape[0] for k,v in cases.items() if k!='labels' and np.asarray(v).ndim]
    n=candidates[0] if candidates else 1
    def field(key,width=1,default=0.):
        value=np.asarray(cases.get(key,default),dtype=np.float64)
        return np.broadcast_to(value, (n,width) if width>1 else (n,)).reshape(n,width)
    mass=np.zeros((n,18)); delta=np.zeros((n,18))
    mass[:,:4]=field('sulfate',4); delta[:,:4]=field('sulfate_delta',4)
    if 'mass' in cases:
        mass=field('mass',18).copy()
    if 'mass_delta' in cases:
        delta=field('mass_delta',18).copy()
    pieces=[field('number',7),field('radius',7,1e-6),field('density',7,1.841),mass,delta]
    for key,default in [('temperature',280),('pressure',101325),('timestep',1),('relative_humidity',.5),('gas',0),('production',0),('cloud_fraction',0),('total_sink',0)]:
        pieces.append(field(key,default=default))
    pieces += [field('modal_sink',7),field('nucleated_number'),field('coagulation',default=1),field('condensation',default=1),field('water',default=1),field('scheme'),field('dry_radius',4,1e-6),field('insoluble_sulfate',3)]
    pieces += [field('organic_scheme'),field('ion_pair_rate'),field('forest_fraction'),field('pbl_level',default=1)]
    inputs=np.concatenate(pieces,axis=1)
    allowed_schemes = [0, 1, 2] if kazil else [0, 1]
    if np.any(~np.isin(inputs[:,76],allowed_schemes)):
        raise ValueError('Only nucleation schemes 0 and 1 are supported; Kazil table unavailable')
    if np.any(~np.isin(inputs[:,75],[0,1])):
        raise ValueError('water must be 0 or 1')
    if operation == 'full' and np.any((inputs[:,74] == 0) & (inputs[:,76] != 0)):
        raise ValueError('Native core condensation=False with nucleation uses an uninitialized sink')
    if repetitions<1 or steps<1:
        raise ValueError('repetitions and steps must be positive')
    text=f'{n} {repetitions} {steps}\n'+'\n'.join(' '.join(format(x,'.17g') for x in row) for row in inputs)+'\n'
    executable=Path(executable).resolve()
    start=time.perf_counter()
    result=subprocess.run([str(executable),str(OPERATIONS.index(operation)+1)],input=text,text=True,capture_output=True,cwd=executable.parent,timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr)
    values=np.fromstring(result.stdout,sep=' ')
    a=values[1:].reshape(n,107)
    outputs={key:a[:,lo:hi].copy() if hi-lo>1 else a[:,lo].copy() for key,lo,hi in [('number',0,7),('sulfate',7,11),('sulfate_delta',11,15),('radius',15,22),('density',22,29),('dry_radius',29,33),('water',33,40),('gas',40,41),('nucleated_number',41,42),('total_sink',42,43),('modal_sink',43,50),('coefficients',50,66),('rate',66,67),('cluster_sulfate',67,68),('mean_particle_mass',68,86),('insoluble_sulfate',86,89),('mass',89,107)]}
    if operation == 'cumulative_normal':
        outputs['cdf'] = outputs['rate'].copy()
        outputs['complement'] = outputs['cluster_sulfate'].copy()
    return outputs, {'native_cpu_seconds':float(values[0]),'wall_seconds':time.perf_counter()-start,'repetitions':repetitions,'steps':steps,'cases':n,'timed_scope':'state reset, native call and output array copy; excludes table loading and input/output serialization'}


def reference_cases():
    number=np.array([[1000,200,30,1,0,0,0],[0,0,0,0,0,0,0],[1e4,1e3,100,10,0,0,0]],float)
    radius=np.broadcast_to([2e-7,2e-6,1e-5,1e-4,2e-6,1e-5,1e-4],number.shape).copy()
    sigma=np.array([1.59,1.59,1.59,2.])
    sulfate=number[:,:4]*(4*np.pi/3)*radius[:,:4]**3*np.exp(4.5*np.log(sigma)**2)*1.841*6.02214179e23/96.0631
    return {'labels':['mixed-pure-sulfate','empty','dense-cold-active'],'number':number,'sulfate':sulfate,'radius':radius,'density':np.full_like(number,1.841),'temperature':np.array([280,298,240.]),'pressure':np.array([101325,101325,70000.]),'relative_humidity':np.array([.5,.8,.7]),'gas':np.array([1e7,0,1e9]),'production':np.array([1e4,0,1e5]),'timestep':np.array([10,1,10.]),'scheme':np.array([0,0,1]),'dry_radius':radius[:,:4].copy()}
