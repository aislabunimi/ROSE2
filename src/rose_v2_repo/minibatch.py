from __future__ import division

import os
import time
import numpy as np
import cv2

import util.layout as lay
import util.postprocessing as post
from object import Segment as sg, ExtendedSegment
from object import Surface as Surface
import rospy
from PIL import Image
import rose_v2_repo.parameters as parameters
import util.disegna as dsg
import util.voronoi as vr
from shapely.geometry.polygon import Polygon
from matplotlib import pyplot as plt
def make_folder(location, folder_name):
    if not os.path.exists(location + '/' + folder_name):
        os.mkdir(location + '/' + folder_name)


def print_parameters(param_obj, path_obj):
    file_parameter = open(path_obj.filepath + 'parameters.txt', 'w')
    param = param_obj.__dict__
    for par in param:
        file_parameter.write(par + ": %s\n" % (getattr(param_obj, par)))


def final_routine(img_ini, param_obj, size, draw, extended_segments_th1_merged, ind, rooms_th1, filepath):
    #post.clear_rooms(filepath + '8b_rooms_th1.png', param_obj, rooms_th1)
    if draw.rooms_on_map:
        segmentation_map_path = dsg.draw_rooms_on_map(img_ini, '8b_rooms_th1_on_map', size, filepath=filepath)
    """
    if draw.rooms_on_map_prediction:
        dsg.draw_rooms_on_map_prediction(img_ini, '8b_rooms_th1_on_map_prediction', size, filepath=filepath)

    if draw.rooms_on_map_lines:
        dsg.draw_rooms_on_map_plus_lines(img_ini, extended_segments_th1_merged, '8b_rooms_th1_on_map_th1' + str(ind),
                                         size,
                                         filepath=filepath)
    """
    # -------------------------------------POST-PROCESSING------------------------------------------------
    #salva una nuova immagine in 8b_rooms_th1_on_map_post, con i colori spreaddati su tutta la mappa e restituisce
    # l'array di colori presenti sulla mappa e il path della nuova mappa appena calcolata.
    segmentation_map_path_post, colors = post.oversegmentation(segmentation_map_path, param_obj.th_post, filepath=filepath)
    return segmentation_map_path_post, colors

class Minibatch:
    def start_main(self, par, param_obj, cleanImg, originalImg, filepath, rooms_voronoi):
        param_obj.tab_comparison = [[''], ['precision_micro'], ['precision_macro'], ['recall_micro'], ['recall_macro'],
                                    ['iou_micro_mean_seg_to_gt'], ['iou_macro_seg_to_gt'], ['iou_micro_mean_gt_to_seg'],
                                    ['iou_macro_gt_to_seg']]

        start_time_main = time.time()
        draw = parameters.ParameterDraw()
        self.rooms_th1 = None
        # ----------------------------1.0_LAYOUT OF ROOMS------------------------------------
        # ------ starting layout
        # read the image with removed non-structural components (obtained using rose)
        orebro_img = cleanImg.copy()
        width = orebro_img.shape[1]
        height = orebro_img.shape[0]
        size = [width, height]
        img_rgb = cv2.bitwise_not(orebro_img)

        # making a copy of original image of occupancy map
        img_ini = originalImg.copy()

        # -------------------------------------------------------------------------------------

        # -----------------------------1.1_CANNY AND HOUGH-------------------------------------

        walls, canny = lay.start_canny_and_hough(img_rgb, param_obj)

        rospy.loginfo("[rose2] walls: {}".format(len(walls)))

        lines = walls
        walls = lay.create_walls(lines)
        rospy.loginfo("[rose2] lines: {} walls: {}".format(len(lines), len(walls)))


        # ------------1.2_SET XMIN, YMIN, XMAX, YMAX OF walls-----------------------------------
        # from all points of walls select x and y coordinates max and min.
        extremes = sg.find_extremes(walls)
        xmin = extremes[0]
        xmax = extremes[1]
        ymin = extremes[2]
        ymax = extremes[3]
        offset = param_obj.offset
        xmin -= offset
        xmax += offset
        ymin -= offset
        ymax += offset

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > size[0]:
            xmax = size[0]
        if ymax > size[1]:
            ymax = size[1]

        # -------------------------------------------------------------------------------------

        # ---------------1.3 EXTERNAL CONTOUR--------------------------------------------------

        img_cont = originalImg.copy()
        (contours, self.vertices) = lay.external_contour(img_cont)


        # -------------------------------------------------------------------------------------

        # ---------------1.4_MEAN SHIFT TO FIND ANGULAR CLUSTERS-------------------------------

        indexes, walls, angular_clusters = lay.cluster_ang(param_obj.h, param_obj.minOffset, walls, diagonals=param_obj.diagonals)

        angular_clusters = lay.assign_orebro_direction(param_obj.comp, walls)

        # -------------------------------------------------------------------------------------

        # ---------------1.5_SPATIAL CLUSTERS--------------------------------------------------

        wall_clusters = lay.get_wall_clusters(walls, angular_clusters)

        wall_cluster_without_outliers = []
        for cluster in wall_clusters:
            if cluster != -1:
                wall_cluster_without_outliers.append(cluster)

        # now that I have a list of clusters related to walls, I want to merge those very close each other
        # obtain representatives of clusters (all except outliers)
        representatives_segments = lay.get_representatives(walls, wall_cluster_without_outliers)

        representatives_segments = sg.spatial_clustering(param_obj.spatialClusteringLineSegmentsThreshold, representatives_segments)

        # now we have a set of Segments with correct spatial cluster, now set the others with same wall_cluster
        spatial_clusters = lay.new_spatial_cluster(walls, representatives_segments, param_obj)

        # -------------------------------------------------------------------------------------

        # ------------------------1.6 EXTENDED_LINES-------------------------------------------

        (self.extended_lines, self.extended_segments) = lay.extend_line(spatial_clusters, walls, xmin, xmax, ymin, ymax)


        self.extended_segments = sg.set_weights(self.extended_segments, walls)
        # this is used to merge together the extended_segments that are very close each other.
        extended_segments_merged = ExtendedSegment.merge_together(self.extended_segments, param_obj.distance_extended_segment, walls)
        extended_segments_merged = sg.set_weights(extended_segments_merged, walls)
        # this is needed in order to maintain the extended lines of the offset STANDARD
        border_lines = lay.set_weight_offset(extended_segments_merged, xmax, xmin, ymax, ymin)
        self.extended_segments_th1_merged, ex_li_removed = sg.remove_less_representatives(extended_segments_merged, param_obj.th1)
        lis = []
        for line in ex_li_removed:
            short_line = sg.create_short_ex_lines(line, walls, size, self.extended_segments_th1_merged)
            if short_line is not None:
                lis.append(short_line)

        lis = sg.set_weights(lis, walls)
        lis, _ = sg.remove_less_representatives(lis, 0.1)
        for el in lis:
            self.extended_segments_th1_merged.append(el)
        #threshold lines (added)
        for e in self.extended_segments_th1_merged:
            if(e.weight < param_obj.th1):
                self.extended_segments_th1_merged.remove(e)

        # -------------------------------------------------------------------------------------

        # --------------------------------1.7_EDGES--------------------------------------------

        # creating edges as intersection between extended lines

        edges = sg.create_edges(self.extended_segments)
        self.edges_th1 = sg.create_edges(self.extended_segments_th1_merged)
        # sg.set_weight_offset_edges(border_lines, edges_th1)

        # -------------------------------------------------------------------------------------

        # ---------------------------1.8_SET EDGES WEIGHTS-------------------------------------

        edges = sg.set_weights(edges, walls)
        self.edges_th1 = sg.set_weights(self.edges_th1, walls)
        # threshold edges (added)
        for e in self.edges_th1:
            if (e.weight < param_obj.threshold_edges):
                self.edges_th1.remove(e)
        # -------------------------------------------------------------------------------------

        # ----------------------------1.9_CREATE CELLS-----------------------------------------
        cells_th1 = Surface.create_cells(self.edges_th1)


        # -------------------------------------------------------------------------------------

        # ----------------Classification of Facces CELLE-----------------------------------------------------
        if self.vertices is None:
            rospy.loginfo('[rose2] ROOMS NOT FOUND')
        else:
            # Identification of Cells/Faces that are Inside or Outside the map
            global centroid
            if par.metodo_classificazione_celle == 1:
                rospy.loginfo("[rose2] 1.classification method: {}".format(par.metodo_classificazione_celle))
                (cells_th1, cells_out_th1, cells_polygons_th1, indexes_th1, cells_partials_th1, contour_th1, centroid_th1, points_th1) = lay.classification_surface(self.vertices, cells_th1, param_obj.division_threshold)
                if len(cells_th1) == 0:
                    rospy.loginfo('[rose2] ROOMS NOT FOUND')
                    return
            # -------------------------------------------------------------------------------------

            # ---------------------------POLYGON CELLS---------------------------------------------
            # TODO this method could be deleted. check. Not used anymore.
            (cells_polygons_th1, polygon_out_th1, polygon_partial_th1, centroid_th1) = lay.create_polygon(cells_th1, cells_out_th1,cells_partials_th1)


            # ----------------------MATRICES L, D, D^-1, ED M = D^-1 * L--------------------------

            (matrix_l_th1, matrix_d_th1, matrix_d_inv_th1, X_th1) = lay.create_matrices(cells_th1, sigma=param_obj.sigma)

            # -------------------------------------------------------------------------------------

            # ----------------DBSCAN PER TROVARE CELLE NELLA STESSA STANZA-------------------------

            cluster_cells_th1 = lay.DB_scan(param_obj.eps, param_obj.minPts, X_th1, cells_polygons_th1)
            #needed for dsg.draw_rooms
            if rooms_voronoi:
                metric_map_path = filepath + 'originalMap.png'
                plt.imsave(metric_map_path, originalImg, cmap='gray')
                colors_th1, fig, ax = dsg.draw_dbscan(cluster_cells_th1, cells_th1, cells_polygons_th1, self.edges_th1,
                                                      contours, '7b_DBSCAN_th1', size, filepath=filepath)

            # -------------------------------------------------------------------------------------

            # ----------------------------POLYGON ROOMS--------------------------------------------

            self.rooms_th1, spaces_th1 = lay.create_space(cluster_cells_th1, cells_th1, cells_polygons_th1)
            rospy.loginfo("[rose2] Number of rooms found: {}".format(len(self.rooms_th1)))
            # -------------------------------------------------------------------------------------

            # searching partial cells
            border_coordinates = [xmin, ymin, xmax, ymax]
            # TODO check how many time is computed

            cells_partials_th1, polygon_partial_th1 = lay.get_partial_cells(cells_th1, cells_out_th1, border_coordinates)

            polygon_out_th1 = lay.get_out_polygons(cells_out_th1)
            #apply voronoi method to find more rooms (it's slower)
            if rooms_voronoi:
                fig, ax, patches = dsg.draw_rooms(self.rooms_th1, colors_th1, '8b_rooms_th1', size, filepath=filepath)
                # ---------------------------------END LAYOUT------------------------------------------
                ind = 0
                segmentation_map_path_post, colors = final_routine(img_ini, param_obj, size, draw,self.extended_segments_th1_merged, ind,
                                                                     self.rooms_th1,
                                                                   filepath=filepath)
                old_colors = []
                voronoi_graph, coordinates = vr.compute_voronoi_graph(metric_map_path, param_obj,
                                                                      False, '', param_obj.bormann, filepath=filepath)
                while old_colors != colors and ind < param_obj.iterations:
                    ind += 1
                    old_colors = colors
                    vr.voronoi_segmentation(patches, colors_th1, size, voronoi_graph, coordinates, param_obj.comp,
                                            metric_map_path, ind, filepath=filepath)
                    segmentation_map_path_post, colors = final_routine(img_ini, param_obj, size, draw,
                                                                       self.extended_segments_th1_merged, ind, self.rooms_th1,
                                                                       filepath=filepath)
                self.rooms_th1 = make_rooms(patches)
                colors_th1 = []
                for r in self.rooms_th1:
                    colors_th1.append(0)
                rospy.loginfo("[rose2] Number of rooms found after voronoi method: {}".format(len(patches)))
                #dsg.draw_rooms(self.rooms_th1, colors_th1, '8b_rooms_th1', size, filepath=filepath)

def make_rooms(patches):
    l = []
    for p in patches:
        l.append(Polygon(p.get_path().vertices))
    return l