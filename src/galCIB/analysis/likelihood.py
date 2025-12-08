# analysis/likelihood.py

# import numpy as np
# import pymaster as nmt
# import pocomc as pc
# from scipy.stats import uniform

# class Sampler:
#     def __init__(self, c_ell_model,
#                  data, cov):

#         """
#         Class to define the sampler.

#         Args:
#             c_ell_model : AnalysisModel class
#             binning_fn : NaMaster binning object
#             data : unbinned, pixel-window and
#                    beam function corrected data dict
#                    with keys containing each measurement
#             cov : unbinned, pixel-window corrected covariance matrix
#         """

#         self.cell_obj = c_ell_model
#         self.data = data
#         self.cov = cov
#         self.invcov = np.linalg.inv(self.cov)

#     def apply_binning(self, b, vec):
#         """
#         Apply bin_cell binning scheme to vector vec
#         """

#         res = b.bin_cell(vec.reshape(1,len(vec)))
#         return res

#     def _compute_binned_pcl_from_cl(self, b):
#         """
#         Returns the binned pseudo-Cl given the
#         true unbinned theory Cl.
#         """

#         # FIXME: pass thetas
#         Cgg, CgI, CII = self.c_ell_model.update_cl()

#         gg_binned = self.apply_binning(b, Cgg)
#         gI_binned = self.apply_binning(b, CgI)
#         II_binned = self.apply_binning(b, CII)


#     # constant part of the Gausisan log-likelihood
#     def _set_logll_norm(self, theta):
#         """
#         Sets the constant normalization term if
#         Gaussian likelihood.
#         """

#         k = len(theta)
#         log_ll_norm = -0.5 * np.log(np.linalg.det(self.cov))
#         log_ll_norm -= k/2 * np.log(2*np.pi)

#         self.log_ll_norm = log_ll_norm

#     def loglike(self, theta):

#         """
#         Returns Gaussian log-likelihood.
#         """

#         tcen, tsat, tprof, tsfr, \
#         tsnu, tir, tsngi, \
#             tsnii, tha = theta_parser(theta)

#         _, cgg, cgI, cII = self.cell_obj.update_cl(theta_cen=tcen,
#                                       theta_sat=tsat,
#                                       theta_prof=tprof,
#                                       theta_sfr=tsfr,
#                                       theta_snu=tsnu,
#                                       theta_IR_hod=tir,
#                                       theta_sn_gI=tsngi,
#                                       theta_sn_II=tsnii,
#                                       hmalpha=tha,
#                                       bin_cl=True)


#         model = np.concatenate((cgg,
#                                 cgI.flatten(),
#                                 cII.flatten()))
#         model_minus_data = model - self.data

#         # Gaussian likelihood
#         log_ll = -0.5* model_minus_data @ self.invcov @ model_minus_data + self.log_ll_norm# Gaussian likelihood

#         #return model
#         return log_ll


#     def log_prior(self, theta):
#         """
#         Returns the prior probability
#         """

#         #FIXME: check if prior definition is properly done in log
#         # versus linear space

#         #FIXME: defining prior by model

#         pc.Prior([

#         ])

#         gamma_cen, log10Mc, sigmaM_cen, Ac, \
#             As, M0, M1, alpha_sat, \
#                 fexp, tau_sat, lambda_NFW, \
#                 etamax, mu0p, mupp, sigmaM0_sfr, tau_sfr, zc, \
#                     _, mu0_Mmin_IR, mup_Mmin_IR, sigma_lnM_IR, \
#                     sg3, sg5, sg8, \
#                     s33, s35, s38, s55, s58, s88, \
#                     hmalpha = theta

#         # SFR conditions
#         inf_cond = ((etamax<0.1)|(etamax>1.0)|\
#                 (mu0p<10)|(mu0p>14)|(mupp<-5)|(mupp>5)|\
#                 (sigmaM0_sfr<0.1)|(sigmaM0_sfr>4)|\
#                 (tau_sfr<0.)|(tau_sfr>1)|(zc<0.5)|(zc>3)|\
#                     (mu0_Mmin_IR<10)|(mu0_Mmin_IR>13)|\
#                     (mup_Mmin_IR<-5)|(mup_Mmin_IR>5)
#                     )

#         # break before needing to check all conditions
#         if inf_cond:
#             return -np.inf

#         # shot-noise conditions
#         inf_cond = (
#             (sg3 < -6)|(sg3 >-1)|(sg5 < -6)|(sg5 >-1)|\
#             (sg8 < -6)|(sg8 >-1)
#         )

#         # break before needing to check all conditions
#         if inf_cond:
#             return -np.inf

#         ## These priors are based on eyeballing
#         # Fig 26 of 2306.06319

#         # Ncen conditions
#         log10M0 = np.log10(M0)

#         inf_cond = ((gamma_cen<1.25)|(gamma_cen>8.5)|\
#                 (log10Mc<11.3)|(log10Mc>11.9)|\
#                 (sigmaM_cen<0.)|(sigmaM_cen>1)
#                 )

#         # break before needing to check all conditions
#         if inf_cond:
#             return -np.inf

#         #Nsat conditions
#         inf_cond = ((As<0.)|(As>1.)|\
#                 (log10M0<10.5)|(log10M0>11.8)|\
#                     (alpha_sat<-0.5)|(alpha_sat>1.5)|\
#                         (fexp<0.)|(fexp>1.)|\
#                             (tau_sat<1)|(tau_sat>11)|\
#                             (lambda_NFW<0.)|(lambda_NFW>1.5)
#                     )


#         # break before needing to check all conditions
#         if inf_cond:
#             return -np.inf

#         return 0 # if inf_cond is False throughout

#     def log_posterior(self, theta):
#         """
#         Returns log of the posterior distribution
#         """

#         lprior = self.log_prior(theta)
#         if lprior == -np.inf: # break early
#             return lprior

#         ll = self.loglike(theta)
#         lpost = ll + lprior

#         return lpost

