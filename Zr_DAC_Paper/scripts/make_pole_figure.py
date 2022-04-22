#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:58:18 2017

@author: joelvbernier
"""
import os

import numpy as np

from matplotlib import pyplot as plt

from hexrd import constants as cnst
from hexrd import instrument
from hexrd.transforms import xfcapi
from hexrd import matrixutil as mutil
from hexrd.xrdutil import EtaOmeMaps
from hexrd import rotations as rot
from hexrd.material import load_materials_hdf5

from skimage.transform import PiecewiseAffineTransform, warp
from skimage import filters
from skimage.morphology import disk

import yaml

pt = PiecewiseAffineTransform()


def dump_variant_data(rmat_list, output_path,
                      rmat_init=np.eye(3), transpose=False):
    rmat_ome = np.empty((len(rmat_list), 3, 3))
    grain_params_list = []
    gw = instrument.GrainDataWriter(output_path)
    for i, rmat in enumerate(rmat_list):
        if transpose:
            rmat_ome[i, :, :] = np.dot(rmat_init, rmat.T)
        else:
            rmat_ome[i, :, :] = np.dot(rmat_init, rmat)
        phi, n = rot.angleAxisOfRotMat(rmat_ome[i, :, :])
        grain_params = np.hstack(
            [phi*n.flatten(), cnst.zeros_3, cnst.identity_6x1]
        )
        gw.dump_grain(i, 1., 0., grain_params)
        grain_params_list.append(grain_params)
    gw.close()
    return grain_params


def zproject_sph_angles(ang_list, chi=0.,
                        method='stereographic',
                        use_mask=False):
    """
    CAVEAT: Z axis projections only!!!
    """
    spts_s = xfcapi.anglesToGVec(ang_list, chi=chi)

    # filter based on hemisphere
    if use_mask:
        pzi = spts_s[:, 2] <= 0
        spts_s = spts_s[pzi, :]
    npts_s = len(spts_s)

    if method.lower() == 'stereographic':
        ppts = np.vstack([
            spts_s[:, 0]/(1. - spts_s[:, 2]),
            spts_s[:, 1]/(1. - spts_s[:, 2]),
            np.zeros(npts_s)
        ]).T
    elif method.lower() == 'equal-area':
        chords = spts_s + np.tile([0, 0, 1], (npts_s, 1))
        scl = np.tile(xfcapi.rowNorm(chords), (2, 1)).T
        ucrd = mutil.unitVector(
                np.hstack([
                        chords[:, :2],
                        np.zeros((len(spts_s), 1))
                ]).T
        )

        ppts = ucrd[:2, :].T * scl
    else:
        raise RuntimeError("method '%s' not recognized" % method)

    if use_mask:
        return ppts, pzi
    else:
        return ppts


def make_wulff_net(ndiv=24, projection='stereographic'):
    """
    TODO: options for generating net boundaries; fixed to Z proj.
    """
    womes = np.radians(
        np.linspace(-1, 1, num=ndiv+1, endpoint=True)*90 - 90
    )
    wetas = np.radians(
        np.linspace(-1, 1, num=ndiv+1, endpoint=True)*90
    )
    pts = []
    for ang in womes:
        net_angs = np.vstack([np.zeros_like(wetas),
                              wetas,
                              ang*np.ones_like(wetas)]).T
        projp = zproject_sph_angles(net_angs, method=projection)
        '''
        This was for conversion to pixel coords
        cpw = (projp[:, 0] + 0.5*oimg_dim)/ps
        rpw = (0.5*oimg_dim - projp[:, 1])/ps
        pts.append(np.vstack([cpw, rpw]).T)
        '''
        pts.append(projp)
    pts = np.dstack(pts)
    return pts


# %% some alpha stuff

mat_dict = load_materials_hdf5('materials.h5')
pd_alpha = mat_dict['alpha_Zr'].planeData
lparms_alpha = pd_alpha.lparms
qsym_alpha = pd_alpha.getQSym()
bmat_alpha = pd_alpha.latVecOps['B']

# Twining
alpha_100_hat = rot.applySym(
    mutil.unitVector(np.dot(bmat_alpha, np.c_[1., 0., 0.].T)), qsym_alpha,
    cullPM=True
)

alpha_101_hat = rot.applySym(
    mutil.unitVector(np.dot(bmat_alpha, np.c_[1., 0., 1.].T)), qsym_alpha,
    cullPM=True
)

alpha_102_hat = rot.applySym(
    mutil.unitVector(np.dot(bmat_alpha, np.c_[1., 0., 2.].T)), qsym_alpha,
    cullPM=True
)

alpha_103_hat = rot.applySym(
    mutil.unitVector(np.dot(bmat_alpha, np.c_[1., 0., 3.].T)), qsym_alpha,
    cullPM=True
)

alpha_112_hat = rot.applySym(
    mutil.unitVector(np.dot(bmat_alpha, np.c_[1., 1., 2.].T)), qsym_alpha,
    cullPM=True
)

# rmat for twinning
# (1 0 -1 1)α ; [-1 0 1 2]α
rmats_ctwin_101 = rot.rotMatOfExpMap(np.pi*alpha_101_hat)

# (1 1 -2 2)α ; [-1 -1 2 3]α
rmats_ctwin_112 = rot.rotMatOfExpMap(np.pi*alpha_112_hat)

# (1 0 -1 2)α ; [-1 0 1 1]α
rmats_ttwin_102 = rot.rotMatOfExpMap(np.pi*alpha_102_hat)

# (1 0 -1 3)
rmats_ctwin_103 = rot.rotMatOfExpMap(np.pi*alpha_103_hat)

gt = np.loadtxt('alpha_Zr_state_0004_grains.out')
gparams_alpha = gt[:, 3:15]
alpha_grains_rmats = rot.rotMatOfExpMap(gt[:, 3:6].T)

# # !!! only for plotting parent alpha pole figures
# eta_ome = EtaOmeMaps('alpha_Zr_pfs_state_00004.npz')
# fig_filename = "parent-101tw_aPF_%s_state_%04d.png"
# gparams = np.atleast_2d(gparams_alpha)
# state_id = 4
# all_vars = alpha_grains_rmats
# colors_alpha = ['r', 'g', 'b']
# marker_omega = np.tile(['d', 'o', 's'], 3)
# # !!! for looking at twins
# marker_omega = np.tile(['d', 'o', 's', 'p', '<', '>'], 3)
# all_vars = [np.dot(i, j.T)
#             for i in alpha_grains_rmats
#             for j in rmats_ttwin_102]

# %%
save_figs = True

state_id = 6  # the state ID form the exp

skip = 10

# eta_ome = EtaOmeMaps('./omega_state_005_omega-zr_maps.npz')
# eta_ome = EtaOmeMaps('results_oZr_0005_omega-zr-5GPa_eta_ome_maps.npz')
# eta_ome = EtaOmeMaps('results_aZr_0000_alpha-zr-0.5GPa_eta_ome_maps.npz')
# eta_ome = EtaOmeMaps('omega_Zr_pfs_state_%05d.npz' % state_id)
eta_ome = EtaOmeMaps('omega_Zr_pfs_state_%05d.npz' % state_id)

instr = instrument.HEDMInstrument(
    yaml.safe_load(open('ge2_hexrd06_py27.yml', 'r'))
)

# # SILCOCK
# sil_quats = np.load('silcock_path_a2o.npy')
# sil_rmats = rot.rotMatOfQuat(sil_quats.T)

# colors_alpha = ['r', 'g', 'b']
# marker_omega = np.tile(['d', 'o', 's'], 3)
# all_vars = [np.dot(i, j.T) for i in alpha_grains_rmats for j in sil_rmats]
# fig_filename = "Silcock_wPF%s_state_%04d.png"
# dump_variant_data(all_vars, os.path.join('./', 'silcock_grains.out'))

# # TAO1 6-var version
# tao1_quats = np.load('tao1_path_a2o.npy')
# tao1_rmats = rot.rotMatOfQuat(tao1_quats.T)

# colors_alpha = ['r', 'g', 'b']
# marker_omega = np.tile(['d', 'o', 's', 'p', '<', '>'], 3)
# all_vars = [np.dot(i, j.T) for i in alpha_grains_rmats for j in tao1_rmats]
# fig_filename = "TAO1-6_wPF%s_state_%04d.png"

# TAO1 12-var version
tao1_quats = np.load('tao1_path_12_a2o.npy')
tao1_rmats = rot.rotMatOfQuat(tao1_quats.T)

colors_alpha = ['r', 'g', 'b']
# marker_omega = np.tile(['d', 'o', 's', 'p', '<', '>'], 3)
marker_omega = np.tile(['d', 'o', 's', 'p', '<', '>',
                        'v', '^', '8', 'h', 'P', 'X'], 3)
all_vars = [np.dot(i, j.T) for i in alpha_grains_rmats for j in tao1_rmats]
fig_filename = "TAO1-12_wPF%s_state_%04d.png"
dump_variant_data(all_vars, os.path.join('./', 'TAO1-12_grains.out'))

# !!! for twins and phase variants
gparams = []
for rmat in all_vars:
    phi, n = rot.angleAxisOfRotMat(rmat)
    gparams.append(
        [np.hstack([phi*n.flatten(), cnst.zeros_3, cnst.identity_6x1])]
    )
gparams = np.vstack(gparams)

simg = instr.simulate_rotation_series(
    eta_ome.planeData, gparams,
    eta_ranges=(0.5*np.pi*np.r_[-1, 1], ),
    ome_ranges=(np.radians([-180, 0]), ),
    ome_period=np.radians([-180, 180]))

master_hkl_ids = [i['hklID'] for i in eta_ome.planeData.hklDataList]

# masks for maps
ome_mask_minus = np.degrees(eta_ome.omegas) < -65
ome_mask_plus = np.degrees(eta_ome.omegas) > 65
eta_mask_minus = np.logical_or(
    np.logical_and(np.degrees(eta_ome.etas) >= -180,
                   np.degrees(eta_ome.etas) < -90),
    np.logical_and(np.degrees(eta_ome.etas) > 90,
                   np.degrees(eta_ome.etas) <= 180))
eta_mask_plus = np.logical_and(np.degrees(eta_ome.etas) >= -90,
                               np.degrees(eta_ome.etas) <= 90)

# labels and net
hkl_str = eta_ome.planeData.getHKLs(asStr=True)
pts = make_wulff_net(ndiv=36, projection='equal-area')

ome_start = [-115, 65]
ome_stop = [-65, 115]
for map_id in range(len(eta_ome.dataStore)):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(r'$\{%s\}_\omega$' % hkl_str[map_id])
    wimgs = []
    for i_pl, (eta_mask, ome_mask) in enumerate(
            zip([eta_mask_plus, eta_mask_minus],
                [ome_mask_minus, ome_mask_plus])):
        img = eta_ome.dataStore[map_id][ome_mask, :][:, eta_mask] + 1
        # img = filters.median(img, footprint=disk(3))
        img = filters.gaussian(img, sigma=cnst.fwhm_to_sigma*1.5)
        nrows_in, ncols_in = img.shape
        two_theta = np.degrees(eta_ome.planeData.getTTh()[map_id])

        tht = 0.5*np.radians(two_theta)
        ctht = np.cos(tht)
        stht = np.sin(tht)

        omes = eta_ome.omegas[ome_mask][::skip]
        etas = eta_ome.etas[eta_mask][::skip]
        op, ep = np.meshgrid(omes,
                             etas,
                             indexing='ij')
        oc, ec = np.meshgrid(range(nrows_in)[::skip],
                             range(ncols_in)[::skip],
                             indexing='ij')
        angs = np.vstack([
            np.radians(two_theta)*np.ones_like(ep.flatten()),
            ep.flatten(),
            op.flatten()
        ]).T

        ppts = zproject_sph_angles(angs, chi=0.0,
                                   method='equal-area', use_mask=False)

        output_dim = max(img.shape)
        oimg_dim = 3.
        ps = oimg_dim / output_dim    # 2*np.sqrt(2)/output_dim
        rp = 0.5*output_dim - ppts[:, 1]/ps
        cp = ppts[:, 0]/ps + 0.5*output_dim

        src = np.vstack([ec.flatten(), oc.flatten(), ]).T
        dst = np.vstack([cp.flatten(), rp.flatten(), ]).T
        tform = pt.estimate(src, dst)

        wimg = warp(
            img,
            inverse_map=pt.inverse,
            output_shape=(output_dim, output_dim)
        )
        wimg[wimg == 0] = np.nan
        wimgs.append(wimg)

        ax[i_pl].imshow(wimg, cmap=plt.cm.bone,
                        vmin=None, vmax=5,
                        interpolation='nearest',
                        extent=0.5*oimg_dim*np.r_[-1, 1, -1, 1])
        for i in range(pts.shape[-1]):
            ax[i_pl].plot(pts[:, 0, i], pts[:, 1, i], 'c-', alpha=0.5, lw=0.25)
            ax[i_pl].plot(pts[i, 0, :], pts[i, 1, :], 'c-', alpha=0.5, lw=0.25)

        ppts_grn = []
        for i, (hkl_id, grn_angs) in enumerate(zip(simg['GE2'][0],
                                                   simg['GE2'][2])):
            # if i not in [0, 3, 5, 6, 9, 10, 17, 24, 28, 29, 34]:
            #     continue
            k = int(np.ceil((i + 1)/(len(all_vars)/3))) - 1
            on_map = np.where(hkl_id == master_hkl_ids[map_id])[0]
            if len(on_map) > 0:
                these_angs = grn_angs[on_map, :]
                these_ppts = zproject_sph_angles(these_angs,
                                                 chi=instr.chi,
                                                 method='equal-area',
                                                 use_mask=False)
                ax[i_pl].plot(these_ppts[:, 0], these_ppts[:, 1],
                              marker_omega[i], ms=12,
                              mec=colors_alpha[k], mfc='none')
        ax[i_pl].axis('off')
        ax[i_pl].set_title(r'$\omega\in(%d^\circ, %d^\circ)$'
                           % (ome_start[i_pl], ome_stop[i_pl]))

    # save figures
    if save_figs:
        fig.savefig(fig_filename
                    % ('_'.join(hkl_str[map_id].replace(' ', '')), state_id),
                    dpi=300)
