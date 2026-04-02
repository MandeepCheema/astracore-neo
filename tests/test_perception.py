"""
AstraCore Neo — Module 5: Perception testbench.

Coverage:
  - Camera: power control, ISP stages, frame shape/dtype, exposure/gain
  - Lidar: power control, scan shape, range filter, ground removal, voxelization, clustering
  - Radar: power control, scan detections, range-Doppler map, CFAR
  - Fusion: extrinsic calibration, lidar→image projection, object fusion
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from perception import (
    BayerPattern,
    CameraConfig,
    CameraSensor,
    ExtrinsicCalib,
    Frame,
    FrameError,
    FusedObject,
    FusionError,
    IntrinsicCalib,
    ISPPipeline,
    ISPStage,
    LidarCluster,
    LidarConfig,
    LidarSensor,
    ObjectClass,
    PerceptionError,
    PixelFormat,
    PointCloud,
    RadarConfig,
    RadarDetection,
    RadarSensor,
    RangeDopplerProcessor,
    SensorFusionEngine,
    VoxelGrid,
    CalibrationError,
    CameraError,
    LidarError,
    RadarError,
    cluster_points,
    filter_range,
    remove_ground,
    voxelize,
)


# ===========================================================================
# Camera tests
# ===========================================================================

class TestCameraPower:
    def test_power_on_off(self):
        cam = CameraSensor()
        assert not cam.is_powered
        cam.power_on()
        assert cam.is_powered
        cam.power_off()
        assert not cam.is_powered

    def test_double_power_on_raises(self):
        cam = CameraSensor()
        cam.power_on()
        with pytest.raises(CameraError):
            cam.power_on()
        cam.power_off()

    def test_double_power_off_raises(self):
        cam = CameraSensor()
        with pytest.raises(CameraError):
            cam.power_off()

    def test_capture_without_power_raises(self):
        cam = CameraSensor()
        with pytest.raises(CameraError):
            cam.capture()


class TestCameraCapture:
    def setup_method(self):
        self.cfg = CameraConfig(width=64, height=48, ai_denoise=False)
        self.cam = CameraSensor(self.cfg)
        self.cam.power_on()

    def teardown_method(self):
        if self.cam.is_powered:
            self.cam.power_off()

    def test_frame_shape(self):
        frame = self.cam.capture(seed=42)
        assert frame.data.shape == (48, 64, 3)

    def test_frame_dtype_float32(self):
        frame = self.cam.capture(seed=1)
        assert frame.data.dtype == np.float32

    def test_frame_values_in_01(self):
        frame = self.cam.capture(seed=7)
        assert float(frame.data.min()) >= 0.0
        assert float(frame.data.max()) <= 1.0

    def test_frame_metadata_id(self):
        frame1 = self.cam.capture(seed=1)
        frame2 = self.cam.capture(seed=2)
        assert frame2.metadata.frame_id == frame1.metadata.frame_id + 1

    def test_frame_counter_increments(self):
        for _ in range(5):
            self.cam.capture()
        assert self.cam.frames_captured == 5

    def test_reproducible_with_seed(self):
        f1 = self.cam.capture(seed=99)
        # Reset counter by creating a new camera
        cam2 = CameraSensor(self.cfg)
        cam2.power_on()
        # Capture once to align frame counter with seed usage
        f2 = cam2.capture(seed=99)
        # Both frames should be identical given same seed
        np.testing.assert_array_equal(f1.data, f2.data)
        cam2.power_off()

    def test_pixel_format(self):
        frame = self.cam.capture()
        assert frame.pixel_format == PixelFormat.RGB888
        assert frame.channels == 3

    def test_frame_dimensions_match_config(self):
        frame = self.cam.capture()
        assert frame.width == self.cfg.width
        assert frame.height == self.cfg.height


class TestISPPipeline:
    def setup_method(self):
        self.cfg = CameraConfig(width=8, height=8, ai_denoise=True, hdr_enabled=False)
        self.isp = ISPPipeline(self.cfg)

    def test_demosaic_shape(self):
        raw = np.zeros((8, 8), dtype=np.uint16)
        rgb = self.isp.demosaic(raw)
        assert rgb.shape == (8, 8, 3)
        assert rgb.dtype == np.float32

    def test_demosaic_wrong_ndim_raises(self):
        raw = np.zeros((8, 8, 3), dtype=np.uint16)
        with pytest.raises(FrameError):
            self.isp.demosaic(raw)

    def test_white_balance_clips_to_01(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
        out = self.isp.white_balance(img, r_gain=3.0, g_gain=1.0, b_gain=3.0)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= 0.0

    def test_gamma_correction_identity_at_gamma1(self):
        cfg2 = CameraConfig(width=4, height=4, gamma=1.0)
        isp2 = ISPPipeline(cfg2)
        img = np.linspace(0, 1, 12, dtype=np.float32).reshape(4, 3, 1)
        out = isp2.gamma_correct(img)
        np.testing.assert_allclose(out, img, atol=1e-6)

    def test_gamma_invalid_raises(self):
        self.cfg.gamma = 0.0
        with pytest.raises(FrameError):
            self.isp.gamma_correct(np.ones((4, 4, 3), dtype=np.float32))

    def test_ai_denoise_passthrough_when_disabled(self):
        cfg_no_denoise = CameraConfig(width=4, height=4, ai_denoise=False)
        isp = ISPPipeline(cfg_no_denoise)
        img = np.random.default_rng(0).random((4, 4, 3)).astype(np.float32)
        out = isp.ai_denoise(img)
        np.testing.assert_array_equal(out, img)

    def test_tone_map_passthrough_when_hdr_disabled(self):
        img = np.ones((4, 4, 3), dtype=np.float32) * 2.0
        out = self.isp.tone_map(img)
        np.testing.assert_array_equal(out, img)

    def test_tone_map_reduces_high_values(self):
        hdr_cfg = CameraConfig(width=4, height=4, hdr_enabled=True)
        isp = ISPPipeline(hdr_cfg)
        img = np.ones((4, 4, 3), dtype=np.float32) * 10.0
        out = isp.tone_map(img)
        assert float(out.max()) < 1.0

    def test_isp_stages_logged(self):
        from perception.camera import FrameMetadata
        import time
        meta = FrameMetadata(
            frame_id=1, timestamp_us=time.monotonic() * 1e6,
            exposure_us=10000.0, analog_gain=1.0, digital_gain=1.0,
            color_temp_k=6500.0,
        )
        raw = np.zeros((8, 8), dtype=np.uint16)
        self.isp.run(raw, meta)
        assert ISPStage.DEMOSAIC.name in meta.isp_stages_applied
        assert ISPStage.WHITE_BAL.name in meta.isp_stages_applied
        assert ISPStage.GAMMA.name in meta.isp_stages_applied


class TestCameraConfig:
    def test_exposure_set_get(self):
        cam = CameraSensor()
        cam.power_on()
        cam.set_exposure(5000.0)
        assert cam._exposure_us == 5000.0
        cam.power_off()

    def test_invalid_exposure_raises(self):
        cam = CameraSensor()
        cam.power_on()
        with pytest.raises(CameraError):
            cam.set_exposure(-100.0)
        cam.power_off()

    def test_analog_gain_bounds(self):
        cam = CameraSensor()
        cam.power_on()
        with pytest.raises(CameraError):
            cam.set_gain(analog=0.5)   # below min
        with pytest.raises(CameraError):
            cam.set_gain(analog=100.0) # above max
        cam.power_off()

    def test_digital_gain_bounds(self):
        cam = CameraSensor()
        cam.power_on()
        with pytest.raises(CameraError):
            cam.set_gain(digital=0.0)
        with pytest.raises(CameraError):
            cam.set_gain(digital=20.0)
        cam.power_off()

    def test_repr(self):
        cam = CameraSensor()
        r = repr(cam)
        assert "CameraSensor" in r
        assert "OFF" in r


# ===========================================================================
# Lidar tests
# ===========================================================================

class TestLidarPower:
    def test_power_on_off(self):
        lidar = LidarSensor()
        assert not lidar.is_powered
        lidar.power_on()
        assert lidar.is_powered
        lidar.power_off()
        assert not lidar.is_powered

    def test_double_power_on_raises(self):
        lidar = LidarSensor()
        lidar.power_on()
        with pytest.raises(LidarError):
            lidar.power_on()
        lidar.power_off()

    def test_scan_without_power_raises(self):
        lidar = LidarSensor()
        with pytest.raises(LidarError):
            lidar.scan()


class TestLidarScan:
    def setup_method(self):
        cfg = LidarConfig(channels=8, angular_resolution_deg=5.0, has_velocity=True)
        self.lidar = LidarSensor(cfg)
        self.lidar.power_on()

    def teardown_method(self):
        if self.lidar.is_powered:
            self.lidar.power_off()

    def test_scan_returns_point_cloud(self):
        cloud = self.lidar.scan(num_points=100, seed=1)
        assert isinstance(cloud, PointCloud)
        assert cloud.num_points == 100

    def test_arrays_equal_length(self):
        cloud = self.lidar.scan(num_points=50, seed=2)
        assert len(cloud.x) == len(cloud.y) == len(cloud.z) == len(cloud.intensity) == 50

    def test_velocity_present_when_enabled(self):
        cloud = self.lidar.scan(num_points=20, seed=3)
        assert cloud.velocity is not None
        assert len(cloud.velocity) == 20

    def test_xyz_returns_n3_array(self):
        cloud = self.lidar.scan(num_points=30, seed=4)
        xyz = cloud.xyz()
        assert xyz.shape == (30, 3)

    def test_xyziv_returns_n5_array(self):
        cloud = self.lidar.scan(num_points=30, seed=5)
        xyziv = cloud.xyziv()
        assert xyziv.shape == (30, 5)

    def test_frame_counter_increments(self):
        for _ in range(3):
            self.lidar.scan(num_points=10)
        assert self.lidar.scans_captured == 3

    def test_mismatched_arrays_raise(self):
        with pytest.raises(LidarError):
            PointCloud(
                x=np.zeros(5), y=np.zeros(3),  # length mismatch
                z=np.zeros(5), intensity=np.zeros(5),
            )


class TestLidarProcessing:
    def setup_method(self):
        rng = np.random.default_rng(42)
        n = 200
        self.cloud = PointCloud(
            x=rng.uniform(-50, 50, n).astype(np.float32),
            y=rng.uniform(-50, 50, n).astype(np.float32),
            z=rng.uniform(-3, 5, n).astype(np.float32),
            intensity=rng.random(n).astype(np.float32),
        )

    def test_filter_range_reduces_points(self):
        filtered = filter_range(self.cloud, min_r=5.0, max_r=30.0)
        assert filtered.num_points < self.cloud.num_points

    def test_filter_range_all_within_bounds(self):
        filtered = filter_range(self.cloud, min_r=0.0, max_r=1000.0)
        assert filtered.num_points == self.cloud.num_points

    def test_filter_range_empty_result(self):
        filtered = filter_range(self.cloud, min_r=999.0, max_r=1000.0)
        assert filtered.num_points == 0

    def test_remove_ground_reduces_points(self):
        ground_cloud = PointCloud(
            x=np.zeros(10, np.float32),
            y=np.zeros(10, np.float32),
            z=np.linspace(-3, 3, 10, dtype=np.float32),
            intensity=np.ones(10, np.float32),
        )
        above = remove_ground(ground_cloud, height_threshold_m=-1.0)
        # Points with z > -1.0 should remain
        assert above.num_points < 10

    def test_voxelize_returns_voxel_grid(self):
        vg = voxelize(self.cloud, voxel_size=5.0)
        assert isinstance(vg, VoxelGrid)
        assert vg.voxels.shape[1] == 3
        assert len(vg.occupancy) == len(vg.voxels)

    def test_voxelize_reduces_points(self):
        vg = voxelize(self.cloud, voxel_size=5.0)
        assert vg.voxels.shape[0] < self.cloud.num_points

    def test_voxelize_empty_cloud(self):
        empty = PointCloud(
            x=np.array([], np.float32), y=np.array([], np.float32),
            z=np.array([], np.float32), intensity=np.array([], np.float32),
        )
        vg = voxelize(empty, voxel_size=1.0)
        assert vg.voxels.shape[0] == 0

    def test_voxelize_invalid_size_raises(self):
        with pytest.raises(LidarError):
            voxelize(self.cloud, voxel_size=0.0)

    def test_cluster_points_returns_list(self):
        clusters = cluster_points(self.cloud, eps=5.0, min_points=3)
        assert isinstance(clusters, list)

    def test_cluster_points_empty_cloud(self):
        empty = PointCloud(
            x=np.array([], np.float32), y=np.array([], np.float32),
            z=np.array([], np.float32), intensity=np.array([], np.float32),
        )
        clusters = cluster_points(empty)
        assert clusters == []

    def test_cluster_has_correct_fields(self):
        # Dense blob — should cluster
        n = 30
        rng = np.random.default_rng(0)
        dense = PointCloud(
            x=rng.uniform(0, 0.5, n).astype(np.float32),
            y=rng.uniform(0, 0.5, n).astype(np.float32),
            z=rng.uniform(0, 0.5, n).astype(np.float32),
            intensity=np.ones(n, np.float32),
        )
        clusters = cluster_points(dense, eps=1.0, min_points=3)
        assert len(clusters) >= 1
        c = clusters[0]
        assert c.centroid.shape == (3,)
        assert c.bbox_min.shape == (3,)
        assert c.bbox_max.shape == (3,)
        assert c.point_count > 0


# ===========================================================================
# Radar tests
# ===========================================================================

class TestRadarPower:
    def test_power_on_off(self):
        radar = RadarSensor()
        assert not radar.is_powered
        radar.power_on()
        assert radar.is_powered
        radar.power_off()
        assert not radar.is_powered

    def test_double_power_on_raises(self):
        radar = RadarSensor()
        radar.power_on()
        with pytest.raises(RadarError):
            radar.power_on()
        radar.power_off()

    def test_scan_without_power_raises(self):
        radar = RadarSensor()
        with pytest.raises(RadarError):
            radar.scan()


class TestRadarScan:
    def setup_method(self):
        cfg = RadarConfig(num_chirps=16, num_samples=64)
        self.radar = RadarSensor(cfg)
        self.radar.power_on()

    def teardown_method(self):
        if self.radar.is_powered:
            self.radar.power_off()

    def test_scan_returns_list(self):
        detections = self.radar.scan(seed=1)
        assert isinstance(detections, list)

    def test_detections_are_radar_detection(self):
        detections = self.radar.scan(seed=2)
        for d in detections:
            assert isinstance(d, RadarDetection)

    def test_detection_range_nonnegative(self):
        detections = self.radar.scan(seed=3)
        for d in detections:
            assert d.range_m >= 0.0

    def test_detection_azimuth_within_fov(self):
        cfg = RadarConfig(num_chirps=16, num_samples=64, azimuth_fov_deg=120.0)
        radar = RadarSensor(cfg)
        radar.power_on()
        detections = radar.scan(seed=5)
        for d in detections:
            assert abs(d.azimuth_deg) <= 60.0
        radar.power_off()

    def test_custom_targets(self):
        targets = [{"range_m": 30.0, "velocity_mps": -10.0, "rcs_linear": 5.0}]
        detections = self.radar.scan(targets=targets, seed=10)
        assert isinstance(detections, list)

    def test_frame_counter_increments(self):
        for _ in range(4):
            self.radar.scan()
        assert self.radar.frames_captured == 4

    def test_repr(self):
        r = repr(self.radar)
        assert "RadarSensor" in r
        assert "77GHz" in r


class TestRangeDopplerProcessor:
    def setup_method(self):
        self.cfg = RadarConfig(num_chirps=16, num_samples=32)
        self.proc = RangeDopplerProcessor(self.cfg)
        self.rng = np.random.default_rng(42)

    def test_rd_map_shape(self):
        targets = [{"range_m": 20.0, "velocity_mps": 5.0, "rcs_linear": 2.0}]
        adc = self.proc._generate_adc_cube(targets, self.rng)
        rd = self.proc.range_doppler_map(adc)
        assert rd.shape == (self.cfg.num_chirps, self.cfg.num_samples)

    def test_rd_map_dtype_float32(self):
        targets = []
        adc = self.proc._generate_adc_cube(targets, self.rng)
        rd = self.proc.range_doppler_map(adc)
        assert rd.dtype == np.float32

    def test_rd_map_nonnegative(self):
        targets = [{"range_m": 10.0, "velocity_mps": 0.0, "rcs_linear": 1.0}]
        adc = self.proc._generate_adc_cube(targets, self.rng)
        rd = self.proc.range_doppler_map(adc)
        assert float(rd.min()) >= 0.0

    def test_cfar_returns_list(self):
        targets = [{"range_m": 10.0, "velocity_mps": 0.0, "rcs_linear": 100.0}]
        adc = self.proc._generate_adc_cube(targets, self.rng)
        rd = self.proc.range_doppler_map(adc)
        dets = self.proc.cfar_detect(rd)
        assert isinstance(dets, list)


# ===========================================================================
# Fusion tests
# ===========================================================================

class TestExtrinsicCalib:
    def test_identity_transform(self):
        calib = ExtrinsicCalib.identity()
        pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = calib.transform(pts)
        np.testing.assert_allclose(out, pts, atol=1e-9)

    def test_translation_only(self):
        calib = ExtrinsicCalib(rotation=np.eye(3), translation=np.array([1.0, 2.0, 3.0]))
        pts = np.array([[0.0, 0.0, 0.0]])
        out = calib.transform(pts)
        np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0], atol=1e-9)

    def test_homogeneous_shape(self):
        calib = ExtrinsicCalib.identity()
        T = calib.homogeneous()
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T, np.eye(4))

    def test_invalid_rotation_shape_raises(self):
        with pytest.raises(CalibrationError):
            ExtrinsicCalib(rotation=np.eye(4), translation=np.zeros(3))

    def test_invalid_translation_shape_raises(self):
        with pytest.raises(CalibrationError):
            ExtrinsicCalib(rotation=np.eye(3), translation=np.zeros(4))


class TestLidarProjection:
    def setup_method(self):
        self.engine = SensorFusionEngine()
        self.intrinsic = IntrinsicCalib(
            fx=500.0, fy=500.0, cx=320.0, cy=240.0,
            width=640, height=480,
        )
        self.lidar_to_cam = ExtrinsicCalib(
            rotation=np.eye(3),
            translation=np.array([0.0, 0.0, 0.0]),
        )

    def test_project_lidar_to_image_shape(self):
        cloud = PointCloud(
            x=np.array([0.0, 1.0, -1.0], np.float32),
            y=np.array([0.0, 0.0, 0.5], np.float32),
            z=np.array([5.0, 10.0, 8.0], np.float32),
            intensity=np.ones(3, np.float32),
        )
        uv = self.engine.project_lidar_to_image(cloud, self.lidar_to_cam, self.intrinsic)
        assert uv.ndim == 2
        assert uv.shape[1] == 2

    def test_project_points_behind_camera_filtered(self):
        cloud = PointCloud(
            x=np.array([1.0, 2.0], np.float32),
            y=np.array([0.0, 0.0], np.float32),
            z=np.array([-5.0, -10.0], np.float32),  # behind camera
            intensity=np.ones(2, np.float32),
        )
        uv = self.engine.project_lidar_to_image(cloud, self.lidar_to_cam, self.intrinsic)
        assert uv.shape[0] == 0

    def test_project_empty_cloud(self):
        empty = PointCloud(
            x=np.array([], np.float32), y=np.array([], np.float32),
            z=np.array([], np.float32), intensity=np.array([], np.float32),
        )
        uv = self.engine.project_lidar_to_image(empty, self.lidar_to_cam, self.intrinsic)
        assert uv.shape[0] == 0


class TestSensorFusion:
    def _make_cluster(self, cx: float, cy: float, cz: float, size: float = 2.0) -> LidarCluster:
        pts = np.array([
            [cx - size, cy - size, cz - size],
            [cx + size, cy + size, cz + size],
            [cx, cy, cz],
        ], dtype=np.float32)
        return LidarCluster(
            cluster_id=1,
            points=pts,
            centroid=np.array([cx, cy, cz], dtype=np.float32),
            bbox_min=pts.min(axis=0),
            bbox_max=pts.max(axis=0),
            point_count=len(pts),
        )

    def test_fuse_lidar_only(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(40.0, 0.0, 0.0)]
        fused = engine.fuse(clusters, [])
        assert len(fused) == 1
        assert isinstance(fused[0], FusedObject)

    def test_fused_object_position_in_ego_frame(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(30.0, 5.0, 1.0)]
        fused = engine.fuse(clusters, [])
        np.testing.assert_allclose(fused[0].position_m, [30.0, 5.0, 1.0], atol=1e-5)

    def test_fuse_with_radar_assigns_velocity(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(40.0, 0.0, 0.0, size=1.5)]
        radar_det = RadarDetection(
            range_m=40.0, azimuth_deg=0.0, elevation_deg=0.0,
            velocity_mps=-10.0, rcs_dbsm=10.0, snr_db=20.0, detection_id=0,
        )
        fused = engine.fuse(clusters, [radar_det])
        assert len(fused) == 1
        assert "radar" in fused[0].sources
        assert fused[0].velocity_mps[0] == pytest.approx(-10.0)

    def test_fuse_multiple_clusters(self):
        engine = SensorFusionEngine()
        clusters = [
            self._make_cluster(20.0, 0.0, 0.0),
            self._make_cluster(50.0, 5.0, 0.0),
            self._make_cluster(80.0, -5.0, 0.0),
        ]
        fused = engine.fuse(clusters, [])
        assert len(fused) == 3

    def test_fused_object_has_class(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(30.0, 0.0, 0.0)]
        fused = engine.fuse(clusters, [])
        assert isinstance(fused[0].object_class, ObjectClass)

    def test_fused_object_confidence_in_01(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(30.0, 0.0, 0.0)]
        fused = engine.fuse(clusters, [])
        assert 0.0 <= fused[0].confidence <= 1.0

    def test_fuse_empty_inputs(self):
        engine = SensorFusionEngine()
        fused = engine.fuse([], [])
        assert fused == []

    def test_object_id_increments(self):
        engine = SensorFusionEngine()
        clusters = [
            self._make_cluster(10.0, 0.0, 0.0),
            self._make_cluster(20.0, 0.0, 0.0),
        ]
        fused = engine.fuse(clusters, [])
        ids = [f.object_id for f in fused]
        assert ids[1] > ids[0]

    def test_reset_counters(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(10.0, 0.0, 0.0)]
        engine.fuse(clusters, [])
        engine.reset_counters()
        fused2 = engine.fuse(clusters, [])
        assert fused2[0].object_id == 1

    def test_fuse_with_camera_frame_adds_source(self):
        from perception.camera import FrameMetadata
        import time
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(30.0, 0.0, 0.0)]
        meta = FrameMetadata(
            frame_id=1, timestamp_us=time.monotonic() * 1e6,
            exposure_us=10000.0, analog_gain=1.0, digital_gain=1.0, color_temp_k=6500.0,
        )
        dummy_frame = Frame(
            data=np.zeros((4, 4, 3), np.float32), metadata=meta,
            width=4, height=4, channels=3, pixel_format=PixelFormat.RGB888,
        )
        fused = engine.fuse(clusters, [], camera_frame=dummy_frame)
        assert "camera" in fused[0].sources

    def test_fused_object_speed(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(40.0, 0.0, 0.0, size=1.5)]
        radar_det = RadarDetection(
            range_m=40.0, azimuth_deg=0.0, elevation_deg=0.0,
            velocity_mps=-15.0, rcs_dbsm=10.0, snr_db=20.0, detection_id=0,
        )
        fused = engine.fuse(clusters, [radar_det])
        assert fused[0].speed_mps == pytest.approx(15.0)

    def test_lidar_source_always_present(self):
        engine = SensorFusionEngine()
        clusters = [self._make_cluster(20.0, 0.0, 0.0)]
        fused = engine.fuse(clusters, [])
        assert "lidar" in fused[0].sources

    def test_extrinsic_offset_applied(self):
        calib = ExtrinsicCalib(
            rotation=np.eye(3),
            translation=np.array([1.0, 0.0, 0.0]),
        )
        engine = SensorFusionEngine(lidar_to_ego=calib)
        clusters = [self._make_cluster(10.0, 0.0, 0.0)]
        fused = engine.fuse(clusters, [])
        # centroid (10, 0, 0) + offset (1, 0, 0) = (11, 0, 0)
        np.testing.assert_allclose(fused[0].position_m[0], 11.0, atol=1e-5)

    def test_repr(self):
        engine = SensorFusionEngine()
        r = repr(engine)
        assert "SensorFusionEngine" in r
