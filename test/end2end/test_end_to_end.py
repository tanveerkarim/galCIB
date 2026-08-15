import os

import numpy as np
import numpy.testing as npt

from galCIB import (
    Survey,
    Cosmology,
    get_hod_model,
    SatProfile,
    SFRModel,
    SnuModel,
    CIBModel,
    PkBuilder,
    AnalysisModel,
)
from galCIB.utils.io import load_my_filters


def test_minimal_pipeline():
    data_dir = os.path.join(".", "data", "minimal")
    filters_dir = os.path.join(data_dir, "filters")
    dndz_file_path = os.path.join(data_dir, "gal", "dndz_extended.p")
    test_data_dir = os.path.join(".", "test", "end2end")
    
    # Galaxy Survey
    dndz_file = np.load(dndz_file_path, allow_pickle=True)
    zs = dndz_file["zrange"]
    pz = dndz_file["dndz"].mean(axis=0)
    zs = np.concatenate((zs, np.arange(2.85, 10.05, 0.1)[2:]))
    pz = np.concatenate((pz, np.arange(2.85, 10.05, 0.1)[2:] * 0))

    # CIB Survey
    nu_obs = [353, 545, 857]  # Planck effective freq. in GHz
    cib_filters = load_my_filters(filters_dir + "/", nu_obs=nu_obs)

    # Survey
    ells = np.arange(100, 2000)
    NSIDE = 1024
    mag_alpha = 2.225
    example_survey = Survey(
        z=zs,
        pz=pz,
        mag_alpha=mag_alpha,
        cib_filters=cib_filters,
        ells=ells,
        nside=NSIDE,
        name="Example",
    )

    # Cosmology
    ks = np.logspace(-3, 1, 500)
    Mh = np.logspace(7, 15, 100)
    cosmo = Cosmology(
        example_survey.z, ks, Mh, colossus_cosmo_name="planck18", use_little_h=False
    )

    example_survey.compute_windows(cosmo)
    # Uncomment to Update Wg and Wmu
    # np.savetxt(os.path.join(test_data_dir, "Wg.txt"), example_survey.Wg)
    # np.savetxt(os.path.join(test_data_dir, "Wmu.txt"), example_survey.Wmu)
    # Check Wg and Wmu
    expected_Wg = np.loadtxt(os.path.join(test_data_dir, "Wg.txt"))
    expected_Wmu = np.loadtxt(os.path.join(test_data_dir, "Wmu.txt"))
    npt.assert_array_equal(example_survey.Wg, expected_Wg)
    npt.assert_array_equal(example_survey.Wmu, expected_Wmu)


    # HODModel
    elg_hod_model = get_hod_model("DESI-ELG", cosmo)

    # SatProfile
    theta_prof = np.array([0.3, 6.14, 1])  # fexp, tau, LambdaNFW
    elg_sat_profile = SatProfile(cosmo, theta_prof, profile_type="mixed")

    # SFRModel
    # Q: Using a different HOD model than above?
    hod_IR = get_hod_model("Zheng05", cosmo)
    sfr_model = SFRModel(name="M21", hod=hod_IR, fsub=0.134)

    # SnuModel
    snu_model_Y23 = SnuModel(name="Y23", cosmo=cosmo, survey=example_survey)

    # CIBModel
    cib_Y23 = CIBModel(sfr_model=sfr_model, snu_model=snu_model_Y23, hod_IR=hod_IR)

    # PkBuilder
    pk_survey = PkBuilder(
        hod_model=elg_hod_model, cib_model=cib_Y23, gal_prof_model=elg_sat_profile, cib_prof_model=elg_sat_profile
    )

    # AnalysisModel
    analysis = AnalysisModel(survey=example_survey, pk3d=pk_survey)


    # Computation
    # gamma, log10Mc, sigmaM, Ac
    theta_cen = np.array([5.29784602, 11.79855648, 0.59689872, 0.1])
    # As, M0, M1, alpha_sat
    theta_sat = np.array([0.72366462, 10**11.77445494, 10**13.0, -0.19832352])
    # fexp, tau, lambda_NFW
    theta_prof = np.array([0.31701458, 5.21965594, 0.08534324])

    # eta_max=1 because Y23 cannot constrain this so the overall norm.
    # goes to L0 in SED
    theta_sfr_Y23 = np.array(
        [
            1,  # eta_max
            11.78,  # log10Mpeak
            2.47,
            0.45,
            1.93,
        ]
    )  # sigmaM0, tau, zc
    # L0, beta_dust, T0, alpha_dust, gamma_dust
    theta_snu_Y23 = np.array([5e-14, 1.98, 21.13, 0.21, 1.7])
    # Mmin, sigma_lnM
    theta_IR_Y23 = np.array([11.47, 0.4])

    # Analysis 
    # Q: Used theta_prof for both theta_gal_prof and theta_cib_prof
    cgg, cgI, cII = analysis.update_cl(
        theta_cen=theta_cen,
        theta_sat=theta_sat,
        theta_gal_prof=theta_prof,
        theta_cib_prof=theta_prof,
        theta_sfr=theta_sfr_Y23,
        theta_snu=theta_snu_Y23,
        theta_IR_hod=theta_IR_Y23,
        theta_sn_gI=np.zeros(3),
        theta_sn_II=np.zeros(6),
        hmalpha=1,
    )

    assert cgg.shape == (1900,)  # len(ells)
    assert cgI.shape == (3, 1900)
    assert cII.shape == (6, 1900)

    # Uncomment to Update cgg, cgI, cII
    # np.savetxt(os.path.join(test_data_dir, "cgg.txt"), cgg)
    # np.savetxt(os.path.join(test_data_dir, "cgI.txt"), cgI)
    # np.savetxt(os.path.join(test_data_dir, "cII.txt"), cII)

    # Check cgg, cgI, cII
    expected_cgg = np.loadtxt(os.path.join(test_data_dir, "cgg.txt"))
    expected_cgI = np.loadtxt(os.path.join(test_data_dir, "cgI.txt"))
    expected_cII = np.loadtxt(os.path.join(test_data_dir, "cII.txt"))

    npt.assert_array_equal(cgg, expected_cgg)
    npt.assert_array_equal(cgI, expected_cgI)
    npt.assert_array_equal(cII, expected_cII)
