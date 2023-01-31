import logging
from typing import List
import pymesh as pm
import networkx as nx
from math import *

from numpy.linalg import norm
import numpy as np

#import wtv.tvtk_curvatures
import scipy.sparse
import sys
# from context import context_manager
import pymesh_constants
from collections import defaultdict


def get_boundary_face_indices(mesh: pm.Mesh):
    mesh.enable_connectivity()

    faces = []
    for index in range(len(mesh.faces)):
        # print(index, " has neighboring triangles: ", mesh.get_face_adjacent_faces(index))
        if len(mesh.get_face_adjacent_faces(index)) < 3:
            faces.append(index)

    return np.array(faces)


def get_edge_lengths(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]

    length1 = np.linalg.norm(v2 - v3, axis=1)
    length2 = np.linalg.norm(v1 - v3, axis=1)
    length3 = np.linalg.norm(v1 - v2, axis=1)

    return np.array([length1, length2, length3]).T


def get_signed_volume(mesh):
    """
    function to get the signed volume of  3D MESH. This is based on the following formula
    Args:
        mesh:

    Returns:

    """
    mesh.add_attribute("face_area")
    mesh.add_attribute("face_centroid")
    mesh.add_attribute("face_normal")

    face_areas = mesh.get_attribute("face_area")
    face_centroids = mesh.get_attribute("face_centroid").reshape((-1, 3))
    face_normals = mesh.get_attribute("face_normal").reshape((-1, 3))
    product_barycenter_normal = np.sum(face_centroids[:, 0] * face_normals[:, 0])  # , axis=1)
    return 1 / 6 * (np.sum(product_barycenter_normal * face_areas))


def get_interior_angles(vertices, faces):
    # numpy (n_triangles, 3) array of edge lengths
    edge_lengths = get_edge_lengths(vertices, faces)

    # Apply cosine law
    a1 = apply_cosine_law(edge_lengths[:, 0], edge_lengths[:, 1], edge_lengths[:, 2])
    a2 = apply_cosine_law(edge_lengths[:, 1], edge_lengths[:, 0], edge_lengths[:, 2])
    a3 = apply_cosine_law(edge_lengths[:, 2], edge_lengths[:, 0], edge_lengths[:, 1])

    return np.rad2deg(np.array([a1, a2, a3]).T)


def apply_cosine_law(a: np.array, b: np.array, c: np.array):
    """
    Args:
        a:
        b:
        c:

    Returns:

    """
    return np.arccos((b**2 + c**2 - a**2) / (2 * b * c))


def getHullStats(hull):
    # convenience function to get hull stats
    hullStats = dict()
    geometry = dict()
    hull.add_attribute(pymesh_constants.FACE_AREA)
    face_areas = hull.get_attribute(pymesh_constants.FACE_AREA)

    hullStats["is_closed"] = hull.is_closed()
    hullStats["is_oriented"] = hull.is_oriented()

    # y is height, x is width and z is depth
    bbox = hull.bbox
    print(bbox)
    geometry["width"] = bbox[1][0] - bbox[0][0]
    geometry["height"] = bbox[1][1] - bbox[0][1]
    geometry["depth"] = bbox[1][2] - bbox[0][2]
    geometry["area"] = np.sum(face_areas)
    geometry["volume"] = hull.volume
    geometry["centroid"] = 0.5 * (bbox[0] + bbox[1])
    hullStats["geometry"] = geometry

    return hullStats


def scaleHull(hull, scale):
    # provide scale with array of x,y,z scale.
    # Example: Scale factor 2 => [2,2,2]
    print("onuronur: ", hull.voxels)
    return pm.form_mesh(hull.vertices * np.array(scale), hull.faces, hull.voxels)


def get_smoothed(mesh, what):
    mesh.enable_connectivity()
    n = len(mesh.vertices)
    result = np.zeros(n)
    for j in range(n):
        r = what[j] + what[mesh.get_vertex_adjacent_vertices(j)].mean()
        r *= 0.5
        result[j] = r
    return result


def filter_small_radii(inds, mesh, percentage=0.9, filter=2):
    from neckpinch import get_all_neighbours

    mesh.enable_connectivity()
    flags = np.zeros(len(mesh.vertices))
    inds0 = np.array(inds)
    flags[inds0] = 1
    result = []
    tot = 0
    for i in inds0:
        neighbours = get_all_neighbours(mesh, i, filter)
        tot += len(neighbours)
        # neighbours = mesh.get_vertex_adjacent_vertices(i)
        a = np.sum(flags[neighbours] / len(neighbours))
        if a > percentage:
            result.append(i)
    logging.info(f"#neighbours={tot / len(inds0)}")
    return result


def get_small_radii(pm_mesh, radius, filter=2, percentage=0.9):
    gauss_vertex = pm_mesh.get_attribute("vertex_gaussian_curvature")
    mean_vertex = pm_mesh.get_attribute("vertex_mean_curvature")
    g = gauss_vertex
    # g = get_smoothed(pm_mesh, g)
    h = mean_vertex * 2
    # h = get_smoothed(pm_mesh, h)
    e = h * h - 4 * g
    bd_edges = pm_mesh.boundary_edges
    bd_vertices = np.unique(bd_edges.ravel())
    n_bounds = len(bd_vertices)
    e[bd_vertices] = 0
    inds = np.where(e < 0)[0]
    # logging.info(f"{h[inds].max()},{g[inds].max()},{g[inds].min()}")
    # logging.info(f"e < 0 : #={len(inds)}")
    e[inds] = 0
    e = np.sqrt(e)
    k1 = 0.5 * (h + e)
    k2 = 0.5 * (h - e)

    ind_gauss_negative = np.where(g < 0)[0]
    negative = len(ind_gauss_negative) - n_bounds
    rest = len(g) - n_bounds
    logging.info(f"percentage of negative curvature {negative / rest}")
    # diff_gauss = np.abs(k1*k2 - g)[ind_gauss_negative]
    # print("diff_gauss", np.quantile(diff_gauss,.9), diff_gauss.mean(), diff_gauss.max(), diff_gauss.min())
    eps = 1e-5
    r1 = 1.0 / (eps + np.abs(k1))
    r2 = 1.0 / (eps + np.abs(k2))
    rmin = np.min((r1, r2), axis=0)
    rmax = np.max((r1, r2), axis=0)
    ind0 = np.where((rmin < radius) & (g < 0) & (rmin < 4 * rmax))[0]
    if filter:
        ind0 = filter_small_radii(ind0, pm_mesh, filter=filter, percentage=percentage)
    return ind0


def not_hyperbolic(pm_mesh):
    bd_edges = pm_mesh.boundary_edges
    bd_vertices = np.unique(bd_edges.ravel())
    g = np.array(pm_mesh.get_attribute("vertex_gaussian_curvature"))
    g[bd_vertices] = 0
    ind_gauss_negative = np.where(g > 1e-3)[0]
    return ind_gauss_negative


def normalize_mesh(vertices, faces):
    centroid = np.mean(vertices, axis=0)
    vertices -= centroid
    radii = norm(vertices, axis=1)
    vertices /= np.amax(radii)
    return pm.form_mesh(vertices, faces)


def print2outershell(what):
    sys.__stdout__.write(f"{what}\n")
    sys.__stdout__.flush()


def integrate(mesh, time_step, L):

    assembler = pm.Assembler(mesh)
    M = assembler.assemble("mass")
    L = -assembler.assemble("graph_laplacian");
    ##Keep L fixed!
    print(L)
    #quit()
    bbox_min, bbox_max = mesh.bbox
    s = np.amax(bbox_max - bbox_min)  # why?
    S = M + (time_step * s) * L

    solver = pm.SparseSolver.create("SparseLU")
    solver.compute(S)
    mv = M * mesh.vertices

    # print2outershell("before solve")
    vertices = solver.solve(mv)
    # print2outershell("after solve")
    return vertices, mesh.faces


# def get_vtk_curvature(mesh):
#     vtk_mesh = wtv.tvtk_curvatures.get_poly_data_surface_with_curvatures(mesh.vertices, mesh.faces)
#     scs = vtk_mesh.point_data.scalars
#     return np.array(scs)


def print_out_stat(what, ar):
    s = "{} mean={} max={} >80%={} >99%={}".format(
        what, np.mean(ar), np.max(ar), np.quantile(ar, 0.8), np.quantile(ar, 0.99)
    )
    logging.info(s)


def _modified_mean_curvature_flow(mesh, L, num_itrs=1, time_step=1e-3):
    bd_edges = mesh.boundary_edges
    bd_vertices = np.unique(bd_edges.ravel())
    mesh = pm.form_mesh(np.copy(mesh.vertices), mesh.faces)

    v0 = mesh.vertices

    bd0 = v0[bd_vertices]
    result = [mesh]
    for i in range(num_itrs):
        try:
            # curvature = get_vtk_curvature(mesh)
            # curvature[bd_vertices] = 0
            # print_out_stat("mean curvature", np.abs(curvature))
            verts, faces = integrate(mesh, time_step, L)  # Keep Laplacian constant to avoid neck pinches
            # print(np.mean(dv), np.max(dv), np.quantile(dv,.8))
            verts[bd_vertices] = bd0
            dv = np.abs(verts - v0)
            print_out_stat("vertex_diff", dv)
            mesh = pm.form_mesh(verts, faces)
            result.append(mesh)
            v0 = mesh.vertices

        except Exception as e:
            # error with ldlt - try again but this time with sparseLU. Does it cause seg-fault?
            logging.error("Mean curvature flow failed at iteration %d with error %s", i, str(e))
            time_step /= 2
            if time_step < 1e-6:
                return mesh
            continue
    return result


def conformalizedMCF2(pm_mesh, num_steps, time_step=5e-3, L=None):
    """
    Calculates the conformlized mean curvature flow in num_step on the mesh (verts,faces)
    :param pm_mesh:
    :param num_steps: int
    :param time_step: float
    :return:
    """
    assembler = pm.Assembler(pm_mesh)
    if L is None:
        L = -assembler.assemble("graph_laplacian")  # Should be Laplacian-Beltrami Matrix
    result = _modified_mean_curvature_flow(pm_mesh, L, num_itrs=num_steps, time_step=time_step)
    mesh = result[-1]
    return mesh, L


# def conformalizedMCF(verts, faces, num_steps, time_step=5e-3, L=None):
#     """
#     Calculates the conformlized mean curvature flow in num_step on the mesh (verts,faces)
#     :param verts: np.array
#     :param faces: np.array
#     :param num_steps: int
#     :param time_step: float
#     :return:
#     """
#     mesh = pm.form_mesh(verts, faces)
#     assembler = pm.Assembler(mesh)
#     if L is None:
#         L = -assembler.assemble("graph_laplacian")  # Should be Laplacian-Beltrami Matrix
#     result = _modified_mean_curvature_flow(mesh, L, num_itrs=num_steps, time_step=time_step)
#     mesh = result[-1]
#     return wtv.tvtk_curvatures.get_poly_data_surface_with_curvatures(mesh.vertices, mesh.faces), L


def translateHull(hull, translation):
    # provide scale with array of x,y,z scale.
    # Example: Scale factor 2 => [2,2,2]

    return pm.form_mesh(hull.vertices + translation, hull.faces, hull.voxels)


def rotateHull(hull, axis, angle, offset):
    offset = np.array(offset)
    axis = np.array(axis)
    angle = radians(angle)
    rot = pm.Quaternion.fromAxisAngle(axis, angle)
    rot = rot.to_matrix()

    vertices = hull.vertices
    bbox = hull.bbox
    centroid = 0.5 * (bbox[0] + bbox[1])
    vertices = np.dot(rot, (vertices - centroid).T).T + centroid + offset

    return pm.form_mesh(vertices, hull.faces, hull.voxels)


# todo remove if possible
def fix_mesh(mesh: pm.Mesh, target_len: float = 0.1) -> pm.Mesh:
    ###target_leng high leads to non-manifold edges
    ###method is too slow to be of much value
    count = 0
    # mesh, __ = pm.remove_degenerated_triangles(mesh, 3)
    # mesh, __ = pm.remove_degenerated_triangles(mesh, 3)
    mesh, __ = pm.collapse_short_edges(mesh, target_len)
    # mesh, __ = pm.remove_degenerated_triangles(mesh, 3)
    mesh, __ = pm.remove_degenerated_triangles(mesh, 3)

    mesh, __ = pm.remove_duplicated_faces(mesh)
    mesh = pm.resolve_self_intersection(mesh)
    # mesh, __ = pm.remove_duplicated_faces(mesh)
    # mesh, __ = pm.remove_duplicated_faces(mesh)
    # mesh = pm.resolve_self_intersection(mesh)
    # print(mesh.faces.shape)
    return mesh


def fix_mesh(mesh, target_len=1.2, num_it=10):
    logging.info("fix_mesh:target resolution: {} mm".format(target_len))
    count = 0
    try:
        mesh, __ = pm.remove_degenerated_triangles(mesh, 100)
    except Exception:
        logging.warning("remove degenerated failed")
    mesh, __ = pm.split_long_edges(mesh, target_len)
    num_vertices = mesh.number_of_vertices
    while True:
        mesh, __ = pm.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pm.collapse_short_edges(mesh, target_len, preserve_feature=True)
        mesh, __ = pm.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.number_of_vertices == num_vertices:
            break
        num_vertices = mesh.number_of_vertices
        logging.info("#v: {}".format(num_vertices))
        count += 1
        if count > num_it:
            break

    mesh = pm.resolve_self_intersection(mesh)
    mesh, __ = pm.remove_duplicated_faces(mesh)
    mesh = pm.compute_outer_hull(mesh)
    mesh, __ = pm.remove_duplicated_faces(mesh)
    mesh, __ = pm.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pm.remove_isolated_vertices(mesh)

    return mesh


def is_valid_mesh(mesh):
    # bounds = mesh.boundary_vertices
    # if len(bounds) > 0:
    #    logging.info("mesh has boundary")
    #    return False
    ints = pm.detect_self_intersection(mesh)
    print("onuronur: ", ints)
    return mesh.is_edge_manifold() and mesh.is_vertex_manifold() and len(ints) == 0

def get_voxel_adjacent_voxels(mesh, index):

    return np.asarray([ind for ind, voxel in enumerate(mesh.voxels) if len(np.intersect1d(mesh.voxels[index], voxel)) == 3])

def get_voxel_adjacent_faces(mesh, index):

    return np.asarray([ind for ind, face in enumerate(mesh.faces) if len(np.intersect1d(mesh.voxels[index], face)) == 3])

def get_face_adjacent_faces(mesh, index):

    return np.asarray([ind for ind, face in enumerate(mesh.faces) if len(np.intersect1d(mesh.faces[index], face)) == 2])

def getVoxelDualGraph(mesh):
    boundaryFaceCount = 0
    importedGraph = nx.Graph(type="skeleton")

    mesh.enable_connectivity()  # needed to avoid crash
    mesh.add_attribute(pymesh_constants.VOXEL_CENTROID)
    mesh.add_attribute(pymesh_constants.FACE_CENTROID)
    voxel_centroids = mesh.get_voxel_attribute(pymesh_constants.VOXEL_CENTROID)
    face_centroids = mesh.get_face_attribute(pymesh_constants.FACE_CENTROID)

    for vc, voxel in enumerate(mesh.voxels):

        adjacentVoxels = mesh.get_voxel_adjacent_voxels(vc)
        boundaryFaces = mesh.get_voxel_adjacent_faces(vc)

        print(mesh.get_face_adjacent_faces(vc))
        print(get_face_adjacent_faces(mesh, vc))
        if vc == 10: quit()


        x1, y1, z1 = voxel_centroids[vc]
        for adjVoxel in adjacentVoxels:
            x2, y2, z2 = voxel_centroids[adjVoxel]
            if not importedGraph.has_node(vc):
                importedGraph.add_node(vc, name=str(vc), x=x1, y=y1, z=z1, tile=False)
            if not importedGraph.has_node(adjVoxel):
                importedGraph.add_node(adjVoxel, name=str(adjVoxel), x=x2, y=y2, z=z2, tile=False)
            if not importedGraph.has_edge(vc, adjVoxel):
                importedGraph.add_edge(vc, adjVoxel)

        boundaryFaces = mesh.get_voxel_adjacent_faces(vc)
        for boundaryFace in boundaryFaces:
            nodeId = mesh.num_voxels + boundaryFaceCount
            xf, yf, zf = face_centroids[boundaryFace]
            importedGraph.add_node(nodeId, name=str(nodeId), x=xf, y=yf, z=zf, tile=False)
            importedGraph.add_edge(vc, nodeId)
            boundaryFaceCount += 1
    return [importedGraph.subgraph(g) for g in nx.connected_components(importedGraph)]


def get_coplanar_faces(mesh: pm.Mesh, faceIndex: int) -> List[int]:
    """find all coplanar connected faces for the face given with "faceIndex"

    Arguments:
        mesh {pm.Mesh} -- The pymesh Mesh containing the faces
        faceIndex {int} -- The index of the reference face

    Returns:
        List[int] -- A list of face indexes that are coplanar with the face given with "faceIndex"
    """

    mesh.enable_connectivity()
    mesh.add_attribute(pymesh_constants.FACE_NORMAL)
    faceNormals1D = mesh.get_attribute(pymesh_constants.FACE_NORMAL)
    faceNormals = np.reshape(faceNormals1D, (len(mesh.faces), -1))

    refNormal = faceNormals[faceIndex] / np.linalg.norm(faceNormals[faceIndex])

    coplanarFaces = set()
    checkedFaces = set()
    uncheckedFaces = set()

    uncheckedFaces.add(faceIndex)

    while len(uncheckedFaces) > 0:
        facesToCheck = set()
        for face in uncheckedFaces:
            if face not in checkedFaces:
                adjFaces = mesh.get_face_adjacent_faces(face)
                for adjFace in adjFaces:
                    adjFaceNormal = faceNormals[adjFace] / np.linalg.norm(faceNormals[adjFace])
                    if np.linalg.norm(adjFaceNormal - refNormal) < 0.01:
                        coplanarFaces.add(adjFace)
                        facesToCheck.add(adjFace)
                    else:
                        checkedFaces.add(adjFace)
                checkedFaces.add(face)
        uncheckedFaces = facesToCheck.copy()
    return list(coplanarFaces)


def get_scalefree_mean_curvature(mesh, scale=None):
    if scale:
        mesh = pm.form_mesh(scale * mesh.vertices, mesh.faces)
    bounds = mesh.boundary_vertices
    print()
    mesh.add_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    h = mesh.get_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    mesh.add_attribute(pymesh_constants.VERTEX_GAUSSIAN_CURVATURE)
    g = mesh.get_attribute(pymesh_constants.VERTEX_GAUSSIAN_CURVATURE)

    h = np.array(h)

    def zero_nan(what):
        which = np.where(np.isnan(what) == True)[0]
        what[which] = 0
        return len(which)

    scaled_k = h / np.sqrt(np.abs(g))
    scaled_k[bounds] = 0
    nnan = zero_nan(scaled_k)
    if nnan:
        logging.info(f"#nans = {nnan}")
    zero_nan(h)
    h[bounds] = 0
    scaled_k[bounds] = 0
    return scaled_k


def get_mean_curvature(mesh):
    mesh.add_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    h = mesh.get_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    bounds = mesh.boundary_vertices
    h = np.array(h)
    h[bounds] = 0

    def zero_nan(what):
        which = np.where(np.isnan(what))[0]
        print("nan ", which)
        what[which] = 0
        return len(which)

    zero_nan(h)
    return np.mean(np.abs(h))


#
# def replace_border_field(pm_mesh, field):
#     bounds = pm_mesh.boundary_vertices
#     pm_mesh.enable_connectivity()
#     for vertex in bounds:
#         neighbours = pm_mesh.get_vertex_adjacent_vertices(vertex)
#         neighbours = [n for n in neighbours if not n in bounds]
#         if len(neighbours) == 1:
#             #n = neighbours[0]
#             field[vertex] = 0#field[n]
#         elif not neighbours:
#             field[vertex] = 0
#         else:
#             pass
#             #field[vertex] = field[neighbours].mean()


def get_mean_vertex_curvature(p):
    p.add_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    mean = np.array(p.get_attribute("vertex_mean_curvature"))
    bounds = p.boundary_vertices
    mean[bounds] = 0
    # replace_border_field(p, mean)
    return np.nan_to_num(mean)


def fix(mesh):
    mesh, __ = pm.remove_duplicated_vertices(mesh)
    mesh, __ = pm.remove_degenerated_triangles(mesh, 1)
    return mesh


def get_narrow_tunnels(mesh: pm.Mesh, min_dist: float, tree=None):
    mesh.add_attribute(pymesh_constants.FACE_CENTROID)
    centroids = mesh.get_face_attribute(pymesh_constants.FACE_CENTROID)
    mesh.add_attribute(pymesh_constants.FACE_NORMAL)
    normals = mesh.get_face_attribute(pymesh_constants.FACE_NORMAL)
    neck_pinch_faces = {}
    if not tree:
        tree = scipy.spatial.cKDTree(centroids, leafsize=11)
    inds = tree.query_ball_point(centroids, min_dist, workers=-1)
    n = len(inds)
    for j in range(n):
        i_normals = normals[inds[j]]
        dots = np.dot(i_normals, i_normals[0])
        opposite = np.argmin(dots)
        if dots[opposite] < -0.9:
            already = None
            for k in inds[j]:
                if k in neck_pinch_faces:
                    already = neck_pinch_faces[k]
            if already is None:
                already = j
            for k in inds[j]:
                neck_pinch_faces[k] = already
    clusters = defaultdict(list)
    for i, k in neck_pinch_faces.items():
        clusters[k].append(i)
    return clusters, tree


def get_narrow_tunnel_vertices(mesh: pm.Mesh, min_dist: float, tree=None):
    clusters, tree = get_narrow_tunnels(mesh, min_dist, tree)
    result = {}
    for k in clusters:
        verts = []
        for face_index in clusters[k]:
            verts.extend(mesh.faces[face_index])
        verts = list(set(verts))
        result[k] = verts
    return result, tree


def mesh2mesh_distance(mesh: pm.Mesh, min_dist: float):
    mesh.add_attribute(pymesh_constants.FACE_CENTROID)
    centroids = mesh.get_face_attribute(pymesh_constants.FACE_CENTROID)
    clusters, tree = get_narrow_tunnels(mesh, min_dist)
    points = []
    logging.info(f"#clusters = {len(clusters)} with width < {min_dist}")
    for k in clusters:
        # print(k, clusters[k])
        points.extend(centroids[clusters[k]])
    return len(clusters), points


def euler(mesh):
    np = len(mesh.vertices)
    nf = len(mesh.faces)
    ne = (3 * nf + len(mesh.boundary_edges)) / 2
    euler = np + nf - ne
    return euler


def genus(mesh):
    return 1 - euler(mesh) / 2


def mesh_statistics(mesh, calc_genus=True, distances=[]):
    # outdated statistics
    # tr = context_manager.ResultsManager()

    # bd_edges = mesh.boundary_edges
    # bd_vertices = np.unique(bd_edges.ravel())
    def zero_nan(what, replace_value=0):
        what[~np.isfinite(what)] = replace_value

    mesh.add_attribute(pymesh_constants.FACE_AREA)
    mesh.add_attribute(pymesh_constants.VERTEX_GAUSSIAN_CURVATURE)
    mesh.add_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE)
    # tr.area_of_smooth_contour = np.sum(mesh.get_attribute(pymesh_constants.FACE_AREA))
    gauss = np.array(mesh.get_attribute(pymesh_constants.VERTEX_GAUSSIAN_CURVATURE))
    zero_nan(gauss)
    bd_edges = mesh.boundary_edges
    bd_vertices = np.unique(bd_edges.ravel())
    gauss[bd_vertices] = 0

    # tr.gaussian_av = np.mean(gauss)
    mean = np.array(mesh.get_attribute(pymesh_constants.VERTEX_MEAN_CURVATURE))
    mean[bd_vertices] = 0
    zero_nan(mean)
    mc = np.abs(mean)
    # tr.absmean_av = np.mean(mc)  # /result["gaussian_integral"]
    # tr.absmean_q0p9 = np.quantile(mc, .9)
    # tr.absmean_q0p5 = np.quantile(mc, .5)
    # if calc_genus:
    # tr.genus = genus(mesh)
    # for r in distances:
    #     anz, points = mesh2mesh_distance(mesh, r)
    #     result[f"anz_narrow_clusters_{r}"] = anz

    gauss_vertex = mesh.get_attribute("vertex_gaussian_curvature")
    g = gauss_vertex
    bd_edges = mesh.boundary_edges
    bd_vertices = np.unique(bd_edges.ravel())
    n_bounds = len(bd_vertices)
    ind_gauss_negative = np.where(g < 0)[0]
    negative = len(ind_gauss_negative) - n_bounds
    rest = len(g) - n_bounds
    # tr.percentage_of_negative_curvature = negative / rest
    # logging.info(f"percentage of negative curvature {tr.percentage_of_negative_curvature}")


# def test_mesh_integrity(mesh):
#     """should be renamed to test_surface"""
#     mesh_statistics(mesh)
#     tr = context_manager.ResultsManager()
#     tr.validation_tests_stats.subtest_self_intersection = pm.detect_self_intersection(mesh)
#     tr.validation_tests_stats.subtest_is_vertex_manifold = mesh.is_vertex_manifold()
#     tr.validation_tests_stats.subtest_is_edge_manifold = mesh.is_edge_manifold()
#     tr.validation_tests_stats.subtest_is_oriented = mesh.is_oriented()


# def test_mesh_printability(mesh, case_config):
#     test_mesh_integrity(mesh)
#     tr = context_manager.ResultsManager()
#     min_dist = case_config.MIN_CHANNEL
#     anz, points = mesh2mesh_distance(mesh, min_dist)
#     tr.print_stats.anz_narrow_clusters = anz
#     tr.print_stats.face_centroids_narrow = points
#     good = True
#     if len(tr.validation_tests_stats.subtest_self_intersection) > 0:
#         good = False
#     if not tr.validation_tests_stats.subtest_has_no_outer_component:
#         good = False
#     for val in (
#         tr.validation_tests_stats.subtest_is_vertex_manifold,
#         tr.validation_tests_stats.subtest_is_edge_manifold,
#         tr.validation_tests_stats.subtest_is_oriented,
#     ):
#         if not val:
#             good = False
#     if (
#         tr.validation_tests_stats.subtest_has_no_degenerated_triangle_angle
#         or tr.validation_tests_stats.subtest_has_no_degenerated_triangle_area
#     ):
#         good = False
#     if tr.print_stats.anz_narrow_clusters:
#         good = False
#     tr.print_stats.is_mesh_ok = good
#     # radius = smoothing_tweaks.MIN_RADIUS_STOP_LAPLACIAN
#     # result["_stop_smoothing_radius"] = mesh.vertices[get_small_radii(mesh, radius)]
#     # radius = smoothing_tweaks.MIN_RADIUS_GROW
#     # radius = 1
#     # result["_grow_radius"] = mesh.vertices[get_small_radii(mesh, radius, percentage=.4, filter=3)]
#     tr.print_stats.non_hyperbolic = mesh.vertices[not_hyperbolic(mesh)]


def split_solid_surface_along_surface(solid_surface, surface):
    """
    Function splits the solid_surface into two parts:
        - faces and vertices that have normals pointing in approximatly the same direction as the surfaces are added to one new mesh
        - faces and vertices that have normals pointing in approximatly the opposite direction as the surface are added to another new mesh
    ONLY WORKS PROPERLY ON BOUNDARY IF SURFACE CUTS THROUGH SOLID SURFACE
    Args:
        solid_surface {PyMesh Mesh}: Printable "3d" triangulated mesh.
        surface {PyMesh Mesh}: "2d" triangulated mesh.

    Returns:
        Both new meshes
    """
    # Compute face normals of meshes
    solid_surface.add_attribute(pymesh_constants.FACE_NORMAL)
    solid_surface.add_attribute(pymesh_constants.FACE_CIRCUMCENTER)

    surface.add_attribute(pymesh_constants.FACE_NORMAL)
    surface.add_attribute(pymesh_constants.FACE_CIRCUMCENTER)

    # Reshape to get dimensions: number of faces x 3
    solid_face_dimensions = np.shape(solid_surface.faces)

    solid_normals = np.reshape(solid_surface.get_attribute(pymesh_constants.FACE_NORMAL), solid_face_dimensions)
    plane_normals = np.reshape(surface.get_attribute(pymesh_constants.FACE_NORMAL), np.shape(surface.faces))
    solid_face_circumcenters = np.reshape(
        solid_surface.get_attribute(pymesh_constants.FACE_CIRCUMCENTER), solid_face_dimensions
    )
    solid_circumcenters = np.reshape(surface.get_attribute(pymesh_constants.FACE_CIRCUMCENTER), np.shape(surface.faces))

    squared_distances, face_indices, closest_points = pm.distance_to_mesh(surface, solid_face_circumcenters)

    top_side_faces = []
    bottom_side_faces = []
    for face in range(solid_face_dimensions[0]):
        vector_solid_surface_center_to_surface_center = (
            solid_face_circumcenters[face] - solid_circumcenters[face_indices[face]]
        )
        normal_scalar_product = (
            vector_solid_surface_center_to_surface_center
            / np.linalg.norm(vector_solid_surface_center_to_surface_center)
        ).dot(plane_normals[face_indices[face]])
        if normal_scalar_product > 0:
            top_side_faces.append(face)
        elif normal_scalar_product < 0:
            bottom_side_faces.append(face)
        else:
            # orthogonal
            pass

    # extract submesh. The 0 means that no faces other than the saved indices are extracted.
    top = pm.submesh(solid_surface, top_side_faces, 0)
    bot = pm.submesh(solid_surface, bottom_side_faces, 0)

    return bot, top


def compute_wall_thickness_at_sample_points(solid_surface, surface, sample_points):
    """
    Args:
        solid_surface {PyMesh Mesh}: Printable "3d" triangulated mesh.
        surface {PyMesh Mesh}: "2d" triangulated mesh.
        sample_points: Numpy List of sample points. Sample points should lay on surface (e.g. vertices, midpoints, ...)
    Returns:
        Numpy List of dimensions: "Number of sample_points" x 1
    """
    bot, top = split_solid_surface_along_surface(solid_surface, surface)

    # I guess it's up for discussion if the wall width should be measured orthogonally to the mid surface or shortest distance
    squared_distances_up, _, _ = pm.distance_to_mesh(top, sample_points)
    squared_distances_down, _, _ = pm.distance_to_mesh(bot, sample_points)

    distances_up = np.sqrt(squared_distances_up)
    distances_down = np.sqrt(squared_distances_down)

    return distances_down + distances_up


def test_wall_thickness(solid_surface, surface):
    sample_points = surface.vertices
    wall_thickness = compute_wall_thickness_at_sample_points(solid_surface, surface, sample_points)
    return wall_thickness


# return ((wall_thickness < min_wall) & (wall_thickness > max_wall)).nonzero()[0]


def filter_normals(normals, mesh):
    from neckpinch import get_all_neighbours

    mesh.enable_connectivity()
    result = np.zeros(normals.shape)
    for i in range(len(result)):
        neighbours = get_all_neighbours(mesh, i, 1)
        a = np.mean(normals[neighbours])
        result[i] = a
    return result


def get_normals_and_curvature(points, faces):
    pm_mesh = pm.form_mesh(points, faces)
    # pm_mesh, info = pm.remove_degenerated_triangles(pm_mesh, num_iterations=2)
    pm_mesh.add_attribute("vertex_normal")
    normals = pm_mesh.get_attribute("vertex_normal")
    normals = normals.reshape(points.shape)
    mesh_statistics(pm_mesh, False)
    # for k in test:
    #     if k[0] == '_':
    #         continue
    #     logging.info(f"{k},{test[k]}")
    # normals = filter_normals(normals, pm_mesh)
    # tr = context_manager.ResultsManager()
    # absmean_q0p5= tr.absmean_q0p5
    absmean_q0p5 = None
    return normals, absmean_q0p5, pm_mesh


def form_mesh(vertices, faces):
    return pm.form_mesh(vertices, faces)


def form_mesh_with_voxels(vertices, faces, voxels):
    return pm.form_mesh(vertices, faces, voxels)


def remove_isolated_vertices(mesh):
    return pm.remove_isolated_vertices(mesh)


def remove_duplicated_faces(mesh, fins_only):
    return pm.remove_duplicated_faces(mesh, fins_only=fins_only)


def remove_duplicated_vertices(mesh, tol=1e-12, importance=None):
    return pm.remove_duplicated_vertices(mesh, tol=tol, importance=importance)


def collapse_short_edges(mesh, eps):
    return pm.collapse_short_edges(mesh, eps)


def remove_obtuse_triangles(mesh, max_angle):
    return pm.remove_obtuse_triangles(mesh, max_angle=max_angle)


def remove_degenerated_triangles(mesh, n):
    return pm.remove_degenerated_triangles(mesh, n)


def load_mesh(path):
    return pm.load_mesh(path)


def save_mesh(filepath, mesh):
    return pm.save_mesh(filepath, mesh)


def tetgen():
    return pm.tetgen()


def detect_self_intersection(mesh):
    return pm.detect_self_intersection(mesh)


def separate_mesh(mesh, connectivity_type="auto"):
    return pm.separate_mesh(mesh, connectivity_type="auto")


def pymesh_mesh():
    return pm.Mesh


# def get_pymesh_stats(mesh):
#     return context_manager.ResultsManager()
