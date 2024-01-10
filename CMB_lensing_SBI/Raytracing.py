# Code by Jeger Broxterman #broxterman@strw.leidenuniv.nl
# cite paper: https://ui.adsabs.harvard.edu/abs/2023arXiv231208450B/abstract

import numpy as np
import healpy as hp
from astropy.cosmology import z_at_value
from astropy import units as u
import unyt
import frogress
import copy
unyt.c.convert_to_units(unyt.km / unyt.s)


class Raytracing:
    def __init__(self, overdensities, cosmology, comoving_edges,nside, NGP = False, volume_weighted = True):
        
        self.delta = overdensities[:]
        self.convergence_raytrace = np.zeros_like(overdensities)
        self.NGP = NGP  # True for determining the quantaties at each shell using the nearest gridpoint, False for bilinear interpolation
        self.nside = nside
        self.npix = int(12 * nside ** 2)
        self.cosmology = cosmology
        self.redshifts = np.array([z_at_value(self.cosmology.comoving_distance, 0.5*(comoving_edges[i]+comoving_edges[i+1]))  for i in range(len(comoving_edges)-1)])
        self.r_mins =  np.array([c.value for c in comoving_edges])[:-1] * unyt.Mpc
        self.r_maxs =  np.array([c.value for c in comoving_edges])[1:] * unyt.Mpc
        self.plane_distances = 0.5*(np.array([c.value for c in comoving_edges])[1:]+np.array([c.value for c in comoving_edges])[:-1])* unyt.Mpc
        self.delta_chis =  (np.array([c.value for c in comoving_edges])[1:]-np.array([c.value for c in comoving_edges])[:-1])* unyt.Mpc

        
        if volume_weighted:
            self.plane_distances = ((self.r_maxs ** 3 + self.r_mins ** 3) / 2) ** (1.0 / 3.0)
            self.redshifts = z_at_value(cosmology.comoving_distance,
    ((self.r_maxs.value ** 3 + self.r_mins.value ** 3) / 2) ** (1.0 / 3.0) * u.Mpc)

        # The xyz of all the pixels
        self.pixels_cart = np.array(hp.pixelfunc.pix2vec(self.nside, range(self.npix), nest=False))  
        # the angles of all the pixels
        self.theta_ini, self.phi_ini = hp.pixelfunc.pix2ang(nside, range(self.npix))
        
        
        # Matrix A_ij (eq 3 https://arxiv.org/pdf/2312.08450.pdf)
        # We need to store 3 planes. The one we want the quantities to be estimated, and the two before that.
        # so mag_matrix will be 2x2x3xnpix.
        
        A_initial = np.array(([1, 0], [0, 1]))  # The initial maginification matrix
        self.mag_matrix = np.zeros((2, 2, 3,  self.npix))  # Magnification matrix (2x2) for all rays (npix) at three (3) planes
        self.mag_matrix[:, :, 0, :] = np.repeat(A_initial[:, :, np.newaxis],  self.npix, axis=2)  # Initialization
        self.mag_matrix[:, :, 1, :] = np.repeat(A_initial[:, :, np.newaxis],  self.npix, axis=2)

        # Matrix B (eq 2 https://arxiv.org/pdf/2312.08450.pdf)
        self.beta_rays = np.zeros((2, self.npix, 3))
        self.beta_rays[0, :, 0], self.beta_rays[1, :, 0] =  self.theta_ini,  self.phi_ini  # Initialization
        self.beta_rays[0, :, 1], self.beta_rays[1, :, 1] =  self.theta_ini,  self.phi_ini

        self.shear_matrix_neighbours = np.zeros((2, 2, 4, self.npix))
        
    def conv_of_ith_shell(self,i):
        unyt.c.convert_to_units(unyt.km / unyt.s)
        conv_value = (
            3 / 2
            * self.cosmology.Om0
            * self.cosmology.H0.value ** 2
            / unyt.c.value ** 2
            * self.plane_distances[i]
            * (1 + self.redshifts[i])
            * self.delta[i]
            * self.delta_chis[i]
        )
        return conv_value.value
    
    def shear_matrix_and_deflection_field_at_i(self, i):
        """
        Determines the deflection field and shear matrix/ tidal tensor/ lensing Jacobian
        of the i-th plane by determining the (first and second order, respectively) covariant
        derivates of the lensing potential.
        """



        convergence = self.conv_of_ith_shell(i,)  # Convergence field for i-th plane
        K_lm = hp.map2alm(convergence)  # Spherical harmonics of the conv field

        lrange, emm = hp.Alm.getlm(lmax = 3*self.nside-1)

        psi_lm = - K_lm * 2 / lrange / (lrange + 1)  # A warning may occur as the first coefficient is undefined 
        psi_lm[0] = 0 # It is set to zero manually after

        _, dtheta, dphi = hp.alm2map_der1(psi_lm, self.nside)  # healpy already scales dphi by sin(theta)

        dtheta_lm = hp.map2alm(dtheta)
        dphi_lm = hp.map2alm(dphi)

        _, dthetadtheta, dthetadphi = hp.alm2map_der1(dtheta_lm, self.nside)
        _, _, dphidphi = hp.alm2map_der1(dphi_lm, self.nside)  # All second order partial derivatives
        return np.array(
            (
                [
                    dthetadtheta,
                    dthetadphi - np.cos(self.theta_ini) / np.sin(self.theta_ini) * dphi,
                    dphidphi + np.cos(self.theta_ini) / np.sin(self.theta_ini) * dtheta,
                ]
            )
        ), np.array(([dtheta, dphi]))

    

    def transport_mag_matrices_neighbours(self,neigbour_positions,mag_matrix):
        """
        Parallel transport the magnification matrix of the 4 neighbouring pixel along the geodesic connecting
        the neigbhour centres to the original ray postition. This is done by rotating the coordinate system
        such that both the initial and neighbour position lay on the equator, in which case the tensor
        components remain constant as it is transported along the sphere (the Chirstoffels vanish), see 
        Becker+2013 CALCLENS appendix A.
        """
        cross = np.cross(neigbour_positions, np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1), axis=0)
        sin = np.linalg.norm(cross, axis=0)  # |axb|=|a||b||sin|
        cross = cross / sin

        angle_mask = sin == 0.
        cross[0,:][angle_mask] = 1.
        cross[1,:][angle_mask] = 0.
        cross[2,:][angle_mask] = 0.
        
        rot_unit_phi_n = (
            np.array(
                (
                    [
                        -neigbour_positions[1, :, :],
                        neigbour_positions[0, :, :],
                        np.zeros((4, self.npix)),
                    ]
                )
            )
            * (
                neigbour_positions * np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)
            ).sum(axis=0)
            + cross
            * (
                np.array(
                    (
                        [
                            -neigbour_positions[1, :, :],
                            neigbour_positions[0, :, :],
                            np.zeros((4, self.npix)),
                        ]
                    )
                )
                * cross
            ).sum(axis=0)
            * (
                1
                - (
                    neigbour_positions * np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)
                ).sum(axis=0)
            )
            + np.cross(
                cross,
                np.array(
                    (
                        [
                            -neigbour_positions[1, :, :],
                            neigbour_positions[0, :, :],
                            np.zeros((4, self.npix)),
                        ]
                    )
                ),
                axis=0,
            )
            * sin
        )

        sin_angle = (
            rot_unit_phi_n
            * np.array(
                (
                    [
                        np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :]
                        * np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :],
                        np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :]
                        * np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :],
                        -(
                            np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :]
                            ** 2
                            + np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :]
                            ** 2
                        ),
                    ]
                )
            )
        ).sum(axis=0) / np.sqrt(
            (1.0 - np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :] ** 2)
            * (1.0 - neigbour_positions[2, :, :] ** 2)
        )
        cos_angle = (
            rot_unit_phi_n
            * np.array(
                (
                    [
                        -np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[1, :, :],
                        np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[0, :, :],
                        np.zeros((4, self.npix)),
                    ]
                )
            )
        ).sum(axis=0) / np.sqrt(
            (1.0 - np.repeat(self.pixels_cart[:, np.newaxis, :], 4, axis=1)[2, :, :] ** 2)
            * (1.0 - neigbour_positions[2, :, :] ** 2)
        )
        rot_unit_phi_n, unit_phi = 0.0, 0.0  # for memory constraints
        rotmat = np.array(
            ([cos_angle, -sin_angle], [sin_angle, cos_angle])
        )  
        return np.einsum("jiml,jkml->ikml", rotmat, np.einsum("ijml,jkml->ikml", mag_matrix, rotmat))

    
    def transport_mag_matrix_NGP(self,positions,mag_matrix):
        """
        Parallel transport the magnification matrix of the NGP pixel along the geodesic connecting
        the NGP centres to the original ray postition. This is done by rotating the coordinate system
        such that both the initial and NGP position lay on the equator, in which case the tensor
        components remain constant as it is transported along the geodesic connecting the centres. 
        See also Becker+2013 (CALCLENS) appendix A
        """
   
        unit_phi = np.array(([-self.pixels_cart[1, :], self.pixels_cart[0, :], np.zeros((self.npix))]))
        
        unit_theta = np.array(([
                    self.pixels_cart[0, :] * self.pixels_cart[2, :],
                    self.pixels_cart[1, :] * self.pixels_cart[2, :],
                    -(self.pixels_cart[0, :] ** 2 + self.pixels_cart[1, :] ** 2),]))

        cross = np.cross(positions, self.pixels_cart, axis=0)
        cos = (positions * self.pixels_cart).sum(axis=0)  # a.b = |a||b|cos
        sin = np.linalg.norm(cross, axis=0)  # axb=|a||b|sin
        cross = cross/ sin
        
        angle_mask = sin == 0.
        cross[0,:][angle_mask] = 1.
        cross[1,:][angle_mask] = 0.
        cross[2,:][angle_mask] = 0.
        


        rotated_basis = np.array(([-positions[1, :], positions[0, :], np.zeros((self.npix))]))
        dot_product = (rotated_basis * cross).sum(axis=0)
        cross_product = np.cross(cross, rotated_basis, axis=0)
        rot_unit_phi_n = (rotated_basis * cos + cross * dot_product * (1 - cos) + cross_product * sin)

        sin_angle = (rot_unit_phi_n * unit_theta).sum(axis=0) / np.sqrt((1.0 - self.pixels_cart[2, :] ** 2) * (1.0 - positions[2, :] ** 2))
        cos_angle = (rot_unit_phi_n * unit_phi).sum(axis=0) / np.sqrt((1.0 - self.pixels_cart[2, :] ** 2) * (1.0 - positions[2, :] ** 2))
        rotmat = np.array(([cos_angle, -sin_angle], [sin_angle, cos_angle]))
       
        
        
        return np.einsum("jil,jkl->ikl", rotmat, np.einsum("ijl,jkl->ikl", mag_matrix, rotmat))

    
    def raytrace_it(self):
        for i_shell in frogress.bar(range(len(self.redshifts)-1)):
            shear_matrix, deflec_field = self.shear_matrix_and_deflection_field_at_i(i_shell)
            # for the first plane, the rays are aimed exactly at the pixel centres
            if i_shell == 0: 
                ray_shear_matrix = np.array(
                    ([shear_matrix[0], shear_matrix[1], shear_matrix[2]])
                )
            # for the other planes we use bilinear interpolation or NGP to estimate the deflection field and shear matrix

            else:   
                ray_coords = self.beta_rays[:, :, 1]
                ray_coords = ray_coords + 2 * np.pi
                ray_coords[1, :] = ray_coords[1, :] % (2 * np.pi)  # phi
                ray_coords[0, :] = ray_coords[0, :] % (np.pi)
                # to make sure all in right range for healpix functions

                if not self.NGP:  # Bilinear interpolation
                    neighbours, weights = hp.pixelfunc.get_interp_weights(
                        self.nside, ray_coords[0, :], ray_coords[1, :], lonlat=False
                    )
                    self.shear_matrix_neighbours[0, 0, :, :] = shear_matrix[0][neighbours]
                    self.shear_matrix_neighbours[1, 0, :, :] = shear_matrix[1][neighbours]
                    self.shear_matrix_neighbours[0, 1, :, :] = shear_matrix[1][neighbours]
                    self.shear_matrix_neighbours[1, 1, :, :] = shear_matrix[2][neighbours]
                    transformed_shear_matrix_neighbours = self.transport_mag_matrices_neighbours(
                        self.pixels_cart[:, neighbours], self.shear_matrix_neighbours
                    )
                    ray_shear_matrix[0, :] = np.sum(
                        weights * transformed_shear_matrix_neighbours[0, 0, :, :], axis=0
                    )
                    ray_shear_matrix[1, :] = np.sum(
                        weights
                        * 0.5
                        * (
                            transformed_shear_matrix_neighbours[0, 1, :, :]
                            + transformed_shear_matrix_neighbours[1, 0, :, :]
                        ),
                        axis=0,
                    )
                    ray_shear_matrix[2, :] = np.sum(
                        weights * transformed_shear_matrix_neighbours[1, 1, :, :], axis=0
                    )
                    deflec_field = np.sum(weights * deflec_field[:, neighbours], axis=1)

                else:
                    closest = hp.pixelfunc.ang2pix( self.nside, ray_coords[0, :], ray_coords[1, :], nest=False, lonlat=False)
                    shear_matrix = np.array(([shear_matrix[0], shear_matrix[1]], [shear_matrix[1], shear_matrix[2]]))

                    transformed_shear_matrix = self.transport_mag_matrix_NGP(self.pixels_cart[:, closest], shear_matrix)
                    ray_shear_matrix[0, :] = transformed_shear_matrix[0, 0, :]
                    ray_shear_matrix[1, :] = (transformed_shear_matrix[0, 1, :] + transformed_shear_matrix[1, 0, :]) / 2
                    ray_shear_matrix[2, :] = transformed_shear_matrix[1, 1, :]
                    deflec_field = deflec_field[:, closest]
                   



            if i_shell ==0: 
                i_shell_ = i_shell + 1
                factor = (self.plane_distances[i_shell_]/ self.plane_distances[i_shell_ + 1]
                    * (self.plane_distances[i_shell_ + 1] - self.plane_distances[i_shell_ - 1])
                    / (self.plane_distances[i_shell_] - self.plane_distances[i_shell_ - 1]))            
            else:
                
                factor = (self.plane_distances[i_shell]/ self.plane_distances[i_shell + 1]
                    * (self.plane_distances[i_shell + 1] - self.plane_distances[i_shell - 1])
                    / (self.plane_distances[i_shell] - self.plane_distances[i_shell - 1]))
            # For recurrance relations
            factor2 = (self.plane_distances[i_shell + 1] - self.plane_distances[i_shell]) / self.plane_distances[i_shell + 1]


            # Compute magnification matrix on next plane for all the rays
            self.mag_matrix[0, 0, 2, :] = (
                (1 - factor) * self.mag_matrix[0, 0, 0, :]
                + factor * self.mag_matrix[0, 0, 1, :]
                - factor2 * (ray_shear_matrix[0, :] * self.mag_matrix[0, 0, 1, :]
                    + ray_shear_matrix[1, :] * self.mag_matrix[1, 0, 1, :]
                )
            )
            self.mag_matrix[1, 0, 2, :] = (
                (1 - factor) * self.mag_matrix[1, 0, 0, :]
                + factor * self.mag_matrix[1, 0, 1, :]
                - factor2
                * (
                    ray_shear_matrix[1, :] * self.mag_matrix[0, 0, 1, :]
                    + ray_shear_matrix[2, :] * self.mag_matrix[1, 0, 1, :]
                )
            )
            self.mag_matrix[0, 1, 2, :] = (
                (1 - factor) * self.mag_matrix[0, 1, 0, :]
                + factor * self.mag_matrix[0, 1, 1, :]
                - factor2
                * (
                    ray_shear_matrix[0, :] * self.mag_matrix[0, 1, 1, :]
                    + ray_shear_matrix[1, :] * self.mag_matrix[1, 1, 1, :]
                )
            )
            self.mag_matrix[1, 1, 2, :] = (
                (1 - factor) * self.mag_matrix[1, 1, 0, :]
                + factor * self.mag_matrix[1, 1, 1, :]
                - factor2
                * (
                    ray_shear_matrix[1, :] * self.mag_matrix[0, 1, 1, :]
                    + ray_shear_matrix[2, :] * self.mag_matrix[1, 1, 1, :]
                )
            )

            self.beta_rays[:, :, 2] = (
                (1 - factor) * self.beta_rays[:, :, 0]
                + factor * self.beta_rays[:, :, 1]
                - factor2 * deflec_field
            )
            

            # Update the amplification matrices, angular coordinates
            self.mag_matrix[:, :, 0, :] = copy.deepcopy(self.mag_matrix[:, :, 1, :])
            self.mag_matrix[:, :, 1, :] = copy.deepcopy(self.mag_matrix[:, :, 2, :])

            self.beta_rays[:, :, 0] = copy.deepcopy(self.beta_rays[:, :, 1])
            self.beta_rays[:, :, 1] = copy.deepcopy(self.beta_rays[:, :, 2])

            self.convergence_raytrace[i_shell] = 1.0 - 0.5 * (self.mag_matrix[0, 0, 2, :] + self.mag_matrix[1, 1, 2, :])
            self.convergence_raytrace[i_shell] -= np.mean(self.convergence_raytrace[i_shell])
