"""
pointclouds : collection of classes and functions to manipulate and save pointclouds
Author : David Rapado
Maintainer : David Rapado

Disclaimer: Some of the code of this script was inspired by or obtained from the pypcd library
"""

import copy
import pickle
import re
import typing
from typing import Tuple
import random

import cv2
import numpy
import open3d
from pytransform3d.transformations import transform, transform_from_pq, vectors_to_points


class PointCloud:
    """Custom pointcloud class that allows saving in PCD format [1]

    [1] https://pcl.readthedocs.io/projects/tutorials/en/latest/pcd_file_format.html

    Parameters
    ----------
    metadata_keys : keys of the metadata values which are stored in the object __dict__. These are:
        - version: Version, usually .7
        - fields: Field names, e.g. ['x', 'y' 'z', 'rgb'].
        - size.`: Field sizes in bytes, e.g. [4, 4, 4, 4].
        - count: Counts per field e.g. [1, 1, 1, 1].
        - width: width of the pointcloud. Equals number of points in unstructured pointclouds
        - height: height of the pointcloud. 1 for unstructured (unorganized) point clouds.
        - viewpoint: A pose for the viewpoint of the cloud, as x y z qw qx qy qz,
                     e.g. [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
        - points: Number of points.
        - type: Data type of each field, e.g. [F, F, F, U].
        - data: Data storage format, ascii or binary.
    points_data : actual pointcloud data, stored in a numpy array
    """

    def __init__(self, metadata: dict, points_data: numpy.array) -> None:
        """Constructor

        Args:
            metadata (dict): dictionary containing the pointcloud metadata
            points_data (numpy.array): actual pointcloud data
        """
        self.metadata_keys = metadata.keys()
        self.__dict__.update(metadata)
        self.points_data = points_data
        self.sanity_check()
        self.has_rgb = True if "rgb" in self.fields else False

    def get_metadata(self) -> dict:
        """Return a copy of the metadata

        Returns:
            dict: dictionary containing a copy of the pointcloud metadata
        """
        metadata = {}
        for k in self.metadata_keys:
            metadata[k] = copy.copy(getattr(self, k))
        return metadata

    @classmethod
    def from_xyz_numpy(
        cls,
        xyz: numpy.array,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Alternative constructor from a pointcloud stored as a numpy array

        Args:
            xyz (numpy.array): actual pointcloud data, (n,3) or (w,h,3)
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Raises:
            ValueError: If the number of dimensions of the input array is smaller than 2 and larger
                        than 3. Input array must be (n,3) or (w,h,3)
            ValueError: If the number of channels of the input array is not 3. You need 3 for XYZ.

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        metadata = {}
        metadata["version"] = 0.7
        dimensions = xyz.shape
        if len(dimensions) == 3:
            width, height, n_channels = dimensions
            xyz = xyz.reshape((-1, n_channels))
        elif len(dimensions) == 2:
            width, n_channels = dimensions
            height = 1
        else:
            raise ValueError("Input array must be (n,3) or (w,h,3)")
        if n_channels != 3:
            raise ValueError("Input array must be (n,3) or (w,h,3)")
        pc_dtype = xyz.dtype
        assert (
            pc_dtype == numpy.float32
        ), f"Pointcloud data type should be float32, but an array with data type {pc_dtype} was given"

        points_data = numpy.rec.fromarrays([xyz[:, 0], xyz[:, 1], xyz[:, 2]], names="x,y,z")

        metadata["fields"] = ["x", "y", "z"]
        metadata["size"] = [4, 4, 4]
        metadata["type"] = ["F", "F", "F"]
        metadata["count"] = [1, 1, 1]
        metadata["width"] = width
        metadata["height"] = height
        metadata["viewpoint"] = viewpoint
        metadata["points"] = width * height
        metadata["data"] = data
        return cls(metadata, points_data)

    @classmethod
    def from_xyzrgb_numpy(
        cls,
        xyz: numpy.array,
        rgb: numpy.array,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Alternative constructor for a pointcloud stored as a numpy array, and colorized
        using a color image

        Args:
            xyz (numpy.array): actual pointcloud data, (n,3) or (w,h,3)
            rgb (numpy.array): color image, (n,3) or (w,h,3)
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".


        Raises:
            ValueError: If the number of dimensions of both input arrays is smaller than 2 and larger
                        than 3. Input array must be (n,3) or (w,h,3)
            ValueError: If the number of channels of both input arrays is not 3. You need 3 for XYZ
                        and 3 for RGB
        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        metadata = {}
        metadata["version"] = 0.7
        dimensions_xyz = xyz.shape
        dimensions_rgb = rgb.shape
        assert dimensions_xyz == dimensions_rgb, "Pointcloud and color arrays must have same size, (n,3) or (w,h,3)"
        if len(dimensions_xyz) == 3:
            width, height, n_channels = dimensions_xyz
            xyz = xyz.reshape((-1, n_channels))
            rgb = rgb.reshape((-1, n_channels))
        elif len(dimensions_xyz) == 2:
            width, n_channels = dimensions_xyz
            height = 1
        else:
            raise ValueError("Pointcloud and color arrays must be (n,3) or (w,h,3)")
        if n_channels != 3:
            raise ValueError("Pointcloud and color arrays must be (n,3) or (w,h,3)")
        pc_dtype = xyz.dtype
        assert (
            pc_dtype == numpy.float32
        ), f"Pointcloud data type should be float32, but an array with data type {pc_dtype} was given"

        rgb_pcl = encode_rgb_for_pcl(rgb)

        points_data = numpy.rec.fromarrays([xyz[:, 0], xyz[:, 1], xyz[:, 2], rgb_pcl], names="x,y,z,rgb")

        metadata["fields"] = ["x", "y", "z", "rgb"]
        metadata["size"] = [4, 4, 4, 4]
        metadata["type"] = ["F", "F", "F", "U"]
        metadata["count"] = [1, 1, 1, 1]
        metadata["width"] = width
        metadata["height"] = height
        metadata["viewpoint"] = viewpoint
        metadata["points"] = width * height
        metadata["data"] = data
        return cls(metadata, points_data)

    @classmethod
    def from_xyz_pkl(
        cls,
        pickle_path: str,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a pointcloud instance from a pickle (.pkl) file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        with open(pickle_path, "rb") as f:
            xyz = pickle.load(f)
        xyz = xyz.astype(numpy.float32)
        return cls.from_xyz_numpy(xyz=xyz, viewpoint=viewpoint, data=data)

    @classmethod
    def from_xyz_pkl_and_color_img(
        cls,
        pickle_path: str,
        rgb_path: str,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a colorized pointcloud instance from a pickle (.pkl) file and a image file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array
            rgb_path (str): path to the image
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        with open(pickle_path, "rb") as f:
            xyz = pickle.load(f)
        xyz = xyz.astype(numpy.float32)
        rgb = cv2.imread(rgb_path)
        return cls.from_xyzrgb_numpy(xyz=xyz, rgb=rgb, viewpoint=viewpoint, data=data)

    @classmethod
    def from_depth_pkl(
        cls,
        pickle_path: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a pointcloud instance from a pickle (.pkl) file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array.
            width (int): camera width, corresponds also with the width of the open3d.visualization.Visualizer window.
            height (int): camera height, corresponds also with the height of the open3d.visualization.Visualizer window.
            fx (float): Camera intrinsics fx
            fy (float): Camera intrinsics fy
            cx (float): Camera intrinsics cx
            cy (float): Camera intrinsics cy
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        with open(pickle_path, "rb") as f:
            depth = pickle.load(f)
        depth = depth.astype(numpy.float32)
        return cls.from_depth(depth, width, height, fx, fy, cx, cy, viewpoint, data)

    @classmethod
    def from_depth(
        cls,
        depth: numpy.array,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a pointcloud instance from a pickle (.pkl) file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array.
            width (int): camera width, corresponds also with the width of the open3d.visualization.Visualizer window.
            height (int): camera height, corresponds also with the height of the open3d.visualization.Visualizer window.
            fx (float): Camera intrinsics fx
            fy (float): Camera intrinsics fy
            cx (float): Camera intrinsics cx
            cy (float): Camera intrinsics cy
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """

        intrinsics = open3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        pcd_o3d_depth = open3d.geometry.PointCloud.create_from_depth_image(
            open3d.geometry.Image(depth), intrinsics, project_valid_depth_only=False
        )
        pcd_numpy_depth = numpy.asarray(pcd_o3d_depth.points).reshape(height, width, 3).astype(numpy.float32)

        return cls.from_xyz_numpy(xyz=pcd_numpy_depth, viewpoint=viewpoint, data=data)

    @classmethod
    def from_depth_pkl_and_color_img(
        cls,
        pickle_path: str,
        rgb_path: str,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a colorized pointcloud instance from a pickle (.pkl) file and a image file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array
            rgb_path (str): path to the image
            width (int): camera width, corresponds also with the width of the open3d.visualization.Visualizer window.
            height (int): camera height, corresponds also with the height of the open3d.visualization.Visualizer window.
            fx (float): Camera intrinsics fx
            fy (float): Camera intrinsics fy
            cx (float): Camera intrinsics cx
            cy (float): Camera intrinsics cy
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        with open(pickle_path, "rb") as f:
            depth = pickle.load(f)
        depth = depth.astype(numpy.float32)
        rgb = cv2.imread(rgb_path)
        return cls.from_depth_and_color_img(depth, rgb, width, height, fx, fy, cx, cy, viewpoint, data)

    @classmethod
    def from_depth_and_color_img(
        cls,
        depth: numpy.array,
        rgb: numpy.array,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Create a colorized pointcloud instance from a pickle (.pkl) file and a image file

        Args:
            pickle_path (str): path to the pickle file which contains the pointcloud array
            rgb_path (str): path to the image
            width (int): camera width, corresponds also with the width of the open3d.visualization.Visualizer window.
            height (int): camera height, corresponds also with the height of the open3d.visualization.Visualizer window.
            fx (float): Camera intrinsics fx
            fy (float): Camera intrinsics fy
            cx (float): Camera intrinsics cx
            cy (float): Camera intrinsics cy
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        intrinsics = open3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        pcd_o3d_depth = open3d.geometry.PointCloud.create_from_depth_image(
            open3d.geometry.Image(depth), intrinsics, project_valid_depth_only=False
        )
        pcd_numpy_depth = numpy.asarray(pcd_o3d_depth.points).reshape(height, width, 3).astype(numpy.float32)
        return cls.from_xyzrgb_numpy(xyz=pcd_numpy_depth, rgb=rgb, viewpoint=viewpoint, data=data)

    @classmethod
    def from_open3d(
        cls,
        pcd: open3d.geometry.PointCloud,
        viewpoint: list = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        data: str = "binary",
    ) -> "PointCloud":
        """Creates a fte_utils PointCloud from an Open3d pointcloud

        Args:
            pcd (open3d.geometry.PointCloud): Open3d pointcloud object
            viewpoint (list, optional): A pose for the viewpoint of the cloud, as x y z qw qx qy qz.
                                        Defaults to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0].
            data (str, optional): Data storage format, ascii or binary. Defaults to "binary".

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        xyz = numpy.asarray(pcd.points).astype(numpy.float32)
        if pcd.has_colors():
            rgb = (numpy.asarray(pcd.colors) * 255).astype(numpy.uint8)
            rgb[:, [0, 2]] = rgb[:, [2, 0]]  # BGR to RGB in (n,3)
            return cls.from_xyzrgb_numpy(xyz=xyz, rgb=rgb, viewpoint=viewpoint, data=data)
        return cls.from_xyz_numpy(xyz=xyz, viewpoint=viewpoint, data=data)

    @classmethod
    def from_pcd(cls, pcd_path: str) -> "PointCloud":
        """Creates a fte_utils PointCloud from an PCD file created by fte_utils.
        PCD files created from other libraries might not load correctly

        Args:
            pcd_path (str): path to the PCD file

        Raises:
            ValueError: when data format for the points is not found or not "binary" or "ascii"

        Returns:
            PointCloud: a fte_utils.pointclouds.PointCloud object
        """
        with open(pcd_path, "rb") as f:
            header = []
            while True:
                ln = f.readline().strip().decode()
                header.append(ln)
                if ln.startswith("DATA"):
                    metadata = parse_header(header)
                    if metadata["fields"] == ["x", "y", "z"]:
                        dtype = numpy.dtype(
                            [
                                ("x", numpy.float32),
                                ("y", numpy.float32),
                                ("z", numpy.float32),
                            ]
                        )
                    elif metadata["fields"] == ["x", "y", "z", "rgb"]:
                        dtype = numpy.dtype(
                            [
                                ("x", numpy.float32),
                                ("y", numpy.float32),
                                ("z", numpy.float32),
                                ("rgb", numpy.uint32),
                            ]
                        )
                    break

            if metadata["data"] == "ascii":
                points_data = parse_ascii_pc_data(f, dtype, metadata)
            elif metadata["data"] == "binary":
                points_data = parse_binary_pc_data(f, dtype, metadata)
            else:
                raise ValueError('DATA field is neither "ascii" or "binary"')
        return cls(metadata, points_data)

    def sanity_check(self) -> None:
        """Check that the metadata is valid

        Raises:
            ValueError: When required metadata fields are missed
        """
        md = self.get_metadata()
        required_fields = (
            "version",
            "fields",
            "size",
            "type",
            "width",
            "height",
            "viewpoint",
            "points",
            "data",
        )
        for f in required_fields:
            if f not in md:
                raise ValueError(f"{f} is required")
        assert (
            len(md["type"]) == len(md["count"]) == len(md["fields"])
        ), "type, count and fields must have the same length"
        assert md["width"] * md["height"] > 0, "width and height must be greater than 0"
        assert md["width"] * md["height"] == md["points"]
        assert len(self.points_data) == md["points"]
        assert md["data"].lower() in ("ascii", "binary")

    def save_ascii(self, fname: str) -> None:
        """Save pointcloud as PCD file in ASCII format

        Args:
            fname (str): path to the pcd output file
        """
        self.save_pcd(fname, "ascii")

    def save_binary(self, fname: str) -> None:
        """Save pointcloud as PCD file in binary format

        Args:
            fname (str): path to the pcd output file
        """
        self.save_pcd(fname, "binary")

    def save_pcd(self, fname: str, data_type: str = None) -> None:
        """Write the pointcloud into a PCD file

        Args:
            fname (str): path to the pcd output file
            data_type(str): data type to save the pointcloud, 'ascii' or 'binary'. Defaults to None
                            and takes the existing value on the pointcloud metadata.
        Raises:
            ValueError: when data type is not 'ascii' or 'binary'
        """
        metadata = self.get_metadata()
        if data_type:
            metadata["data"] = data_type

        with open(fname, "w") as f:
            header = pcd_metadata_to_str(metadata)
            f.write(header)

        with open(fname, "ab") as f:
            if metadata["data"].lower() == "ascii":
                fmtstr = build_pcd_ascii_fmtstr(self)
                numpy.savetxt(f, self.points_data, fmt=fmtstr)
            elif metadata["data"].lower() == "binary":
                f.write(self.points_data.tobytes("C"))
            else:
                raise ValueError("unknown DATA type")

    def copy(self) -> "PointCloud":
        """Create a copy of the pointcloud object

        Returns:
            PointCloud: new pointcloud instance with copied data
        """
        new_pc_data = numpy.copy(self.points_data)
        new_metadata = self.get_metadata()
        return self.__class__(new_metadata, new_pc_data)

    def get_points_xyz_array(self, reshape: bool = False) -> numpy.ndarray:
        """Returns the points of the pointcloud as a numpy array.

        Args:
            reshape (bool, optional): If true, it reshapes the pointcloud into its structured shape. Defaults to False.

        Returns:
            numpy.ndarray: output numpy array containing the points of the pointcloud.
        """
        xyz = numpy.array(
            [
                self.points_data["x"],
                self.points_data["y"],
                self.points_data["z"],
            ]
        ).T
        if reshape:
            xyz = numpy.reshape(xyz, (self.width, self.height, 3))
        return xyz

    def get_points_rgb_xyz_array(self, reshape: bool = False) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Returns the points and the RGB data of the pointcloud as a numpy array.

        Args:
            reshape (bool, optional): If true, it reshapes the pointcloud into its structured shape. Defaults to False.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: a tuple containing:
                - numpy.ndarray: output numpy array containing the RGB data of the pointcloud.
                - numpy.ndarray: output numpy array containing the points of the pointcloud.
        """

        xyz = numpy.array(
            [
                self.points_data["x"],
                self.points_data["y"],
                self.points_data["z"],
            ]
        ).T
        rgb = decode_rgb_from_pcl(self.points_data["rgb"])
        if reshape:
            xyz = numpy.reshape(xyz, (self.width, self.height, 3))
            rgb = numpy.reshape(rgb, (self.width, self.height, 3))
        return rgb, xyz

    def get_tranformation_matrix(self) -> numpy.ndarray:
        return transform_from_pq(self.viewpoint)

    def transform_to_origin(self) -> None:
        """Transform a pointcloud using its viewpoint metadata. This converts the viewpoint
        to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        """
        T = self.get_tranformation_matrix()
        xyz = self.get_points_xyz_array()
        vectors = vectors_to_points(xyz)
        xyz_t = transform(T, vectors)
        self.points_data["x"] = xyz_t[:, 0]
        self.points_data["y"] = xyz_t[:, 1]
        self.points_data["z"] = xyz_t[:, 2]
        self.viewpoint = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def get_copy_transformed_to_origin(self) -> "PointCloud":
        """Create a copy of the pointcloud object, but transformed using its viewpoint
        metadata. This converts the viewpoint to [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

        Returns:
            PointCloud: new pointcloud instance with copied data
        """
        new_pc_data = numpy.copy(self.points_data)
        new_metadata = self.get_metadata()
        T = transform_from_pq(new_metadata.viewpoint)
        xyz = numpy.array(
            [
                new_pc_data["x"],
                new_pc_data["y"],
                new_pc_data["z"],
            ]
        ).T
        vectors = vectors_to_points(xyz)
        xyz_t = transform(T, vectors)
        new_pc_data["x"] = xyz_t[:, 0]
        new_pc_data["y"] = xyz_t[:, 1]
        new_pc_data["z"] = xyz_t[:, 2]
        new_metadata.viewpoint = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        return self.__class__(new_metadata, new_pc_data)

    def get_open3d(self) -> open3d.geometry.PointCloud:
        """Get the pointcloud stored as an open3d object

        Returns:
            open3d.geometry.PointCloud: open3d pointcloud
        """
        pcd_o3d = open3d.geometry.PointCloud()
        if self.has_rgb:
            xyz, rgb = self.get_points_rgb_xyz_array()
            pcd_o3d.points = open3d.utility.Vector3dVector(xyz)
            rgb = (rgb / 255).astype(numpy.float32)
            pcd_o3d.colors = open3d.utility.Vector3dVector(rgb)
        else:
            xyz = self.get_points_xyz_array()
            pcd_o3d.points = open3d.utility.Vector3dVector(xyz)
        return pcd_o3d

    def get_pcd_removed_non_finite_points(self, remove_nan: bool = True, remove_infinite: bool = True) -> "PointCloud":
        """Get a pcd with removed non finite points. It uses open3d.

        Args:
            remove_nan (bool, optional): Remove nans. Defaults to True.
            remove_infinite (bool, optional): Remove infinites. Defaults to True.

        Returns:
            PointCloud: a PointCloud object without non finite points.
        """
        pcd = self.get_open3d()
        pcd.remove_non_finite_points(remove_nan=remove_nan, remove_infinite=remove_infinite)
        return PointCloud.from_open3d(pcd, self.viewpoint, self.data)


def build_pcd_ascii_fmtstr(pc: PointCloud) -> list:
    """Make a format string for printing to ascii.

    Args:
        pc (PointCloud): a fte_utils.pointclouds.PointCloud object

    Raises:
        ValueError: when a dimension type is not known.

    Returns:
        list: list of string formatting to save pointcloud data in ASCII format
    """
    format_string = []
    for t, cnt in zip(pc.type, pc.count):
        if t == "F":
            format_string.extend(["%.10f"] * cnt)
        elif t == "I":
            format_string.extend(["%d"] * cnt)
        elif t == "U":
            format_string.extend(["%u"] * cnt)
        else:
            raise ValueError(f"unknown type {t}")
    return format_string


def pcd_metadata_to_str(metadata: dict) -> str:
    """Given metadata as dictionary, return a string version of it so
    it can be written into a .pcd file.

    Args:
        metadata (dict): dictionary containing a pointcloud metadata.

    Returns:
        str: pointcloud metadata in string format .
    """

    template = """\
VERSION {version}
FIELDS {fields}
SIZE {size}
TYPE {type}
COUNT {count}
WIDTH {width}
HEIGHT {height}
VIEWPOINT {viewpoint}
POINTS {points}
DATA {data}
"""
    str_metadata = metadata.copy()
    new_fields = []
    for f in metadata["fields"]:
        if f == "_":
            new_fields.append("padding")
        else:
            new_fields.append(f)
    str_metadata["fields"] = " ".join(new_fields)
    str_metadata["size"] = " ".join(map(str, metadata["size"]))
    str_metadata["type"] = " ".join(metadata["type"])
    str_metadata["count"] = " ".join(map(str, metadata["count"]))
    str_metadata["width"] = str(metadata["width"])
    str_metadata["height"] = str(metadata["height"])
    str_metadata["viewpoint"] = " ".join(map(str, metadata["viewpoint"]))
    str_metadata["points"] = str(metadata["points"])
    tmpl = template.format(**str_metadata)
    return tmpl


def encode_rgb_for_pcl(rgb: numpy.array) -> numpy.array:
    """Encode (n, 3) RGB data into the 32 bit-packed format used by PCL

    Args:
        rgb (numpy.array): rgb data stored in a (n,3) uint8 array.

    Returns:
        numpy.array: 32 bit-packed rgb data stored in a (n,) uint32 array
    """
    red = rgb[:, 0].astype(numpy.uint32)
    green = rgb[:, 1].astype(numpy.uint32)
    blue = rgb[:, 2].astype(numpy.uint32)

    rgb = (numpy.left_shift(red, 16) + numpy.left_shift(green, 8) + numpy.left_shift(blue, 0)).T

    return rgb


def decode_rgb_from_pcl(rgb: numpy.array) -> numpy.array:
    """Decode the 32 bit-packed format used by PCL into a standard (n, 3) RGB data into

    Args:
        rgb (numpy.array): 32 bit-packed rgb data stored in a (n,) uint32 array

    Returns:
        numpy.array: rgb data stored in a (n,3) uint8 array
    """
    rgb.dtype = numpy.uint32

    rgb_arr = numpy.array(
        [
            numpy.right_shift(rgb, 16).astype(numpy.uint8),
            numpy.right_shift(rgb, 8).astype(numpy.uint8),
            numpy.right_shift(rgb, 0).astype(numpy.uint8),
        ]
    ).T

    return rgb_arr


def parse_binary_pc_data(f: typing.IO, dtype: numpy.dtype, metadata: dict) -> numpy.array:
    """Use numpy to parse (read) ascii pointcloud data

    Args:
        f (typing.IO): file object created using `open(path, "r")`
        dtype (numpy.dtype): dtype for structured numpy array
        metadata (dict): PCD metadata

    Returns:
        numpy.array: structured numpy array containing the pointcloud data
    """
    rowstep = metadata["points"] * dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return numpy.fromstring(buf, dtype=dtype)


def parse_ascii_pc_data(f: typing.IO, dtype: numpy.dtype, metadata: dict) -> numpy.array:
    """Use numpy to parse (read) ascii pointcloud data

    Args:
        f (typing.IO): file object (binary) created using `open(path, "rb")`
        dtype (numpy.dtype): dtype for structured numpy array
        metadata (dict): PCD metadata

    Returns:
        numpy.array: structured numpy array containing the pointcloud data
    """
    return numpy.loadtxt(f, dtype=dtype, delimiter=" ")


def parse_header(lines: list) -> dict:
    """Parse header of PCD files

    Args:
        lines (list[str]): list of lines containing the header of the PCD file

    Returns:
        dict: PCD metadata
    """
    metadata = {}
    for line in lines:
        match = re.match("(\w+)\s+(.+)", line)
        key, value = match.group(1).lower(), match.group(2)
        # key, value = line.lower().split(" ")
        if key == "version":
            metadata[key] = value
        elif key in ("fields", "type"):
            metadata[key] = value.split()
        elif key in ("size", "count"):
            metadata[key] = list(map(int, value.split()))
        elif key in ("width", "height", "points"):
            metadata[key] = int(value)
        elif key == "viewpoint":
            metadata[key] = list(map(float, value.split()))
        elif key == "data":
            metadata[key] = value.strip().lower()
    # add some reasonable defaults
    if "count" not in metadata:
        metadata["count"] = [1] * len(metadata["fields"])
    if "viewpoint" not in metadata:
        metadata["viewpoint"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if "version" not in metadata:
        metadata["version"] = "0.7"
    return metadata


def add_gaussian_noise(pcd: numpy.array, k_min: float, k_max: float) -> numpy.array:
    """Adds gaussian noise to a pointcloud. It generates a noise array using numpy.random.randn,
    then it scales it down using a scale factor. This scale factor is chosen randomly between a
    minimum and a maximum number.

    Args:
        pcd (numpy.array): A pointcloud, it supports organized (w, h, n_c) or unorganized (n_p, n_c)
            pointclouds.
        k_min (float): Minimum scaling factor of the noise. It affects the noise size.
        k_max (float): Maximum scaling factor of the noise. It affects the noise size.

    Returns:
        numpy.array: A pointcloud with added noise, with the same shape as the input pointcloud.
    """
    noise = (numpy.random.randn(*pcd.shape) * random.uniform(k_min, k_max)).astype(numpy.float32)
    return pcd + noise


def add_outlier_noise(pcd: numpy.array, outlier_prob: float) -> numpy.array:
    """Add outliers noise to a pointcloud. Only un-structured pointclouds are supported.

    Args:
        pcd (numpy.array): Un-structured (Nx3 or Nx6) input pointcloud.
        outlier_prob (float): Probability of outliers. Between 0 and 1, where 0 means no outliers
            and 1 means all outliers.

    Returns:
        numpy.array: The resulting noisy pointcloud.
    """
    assert outlier_prob > 0 and outlier_prob < 1, "outlier_prob should be between 0 and 1."
    if numpy.isnan(pcd).any():
        print("Pointcloud contains NaNs. Removing them")
        pcd = pcd[~numpy.isnan(pcd).any(axis=1)]
    assert (
        len(pcd.shape) == 2
    ), f"Only un-structured pointclouds are supported, but a pointcloud with shape {pcd.shape} was passed."
    min_x = pcd[:, 0].min()
    min_y = pcd[:, 1].min()
    min_z = pcd[:, 2].min()
    max_x = pcd[:, 0].max()
    max_y = pcd[:, 1].max()
    max_z = pcd[:, 2].max()
    n_p, n_c = pcd.shape
    noise_x = numpy.random.uniform(low=min_x, high=max_x, size=(n_p, 1))
    noise_y = numpy.random.uniform(low=min_y, high=max_y, size=(n_p, 1))
    noise_z = numpy.random.uniform(low=min_z, high=max_z, size=(n_p, 1))
    noise = numpy.concatenate((noise_x, noise_y, noise_z), axis=1)
    probabilities = numpy.random.random_sample(size=(n_p, 1))
    filter_ = probabilities > (1 - outlier_prob)
    pcd[filter_[:, 0], :3] = noise[filter_[:, 0], :3]
    return pcd
