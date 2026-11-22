import unittest

from tools.pose.temporal import TemporalAngles


class TemporalTests(unittest.TestCase):
    def engine(self,**overrides):
        options=dict(lag_seconds=(.1,),mean_seconds=(.2,),mean_lag_seconds=.1,
                     max_gap_seconds=.11,lag_tolerance_seconds=.02)
        options.update(overrides)
        return TemporalAngles(['angle'],**options)

    def update(self,engine,t,value=0,video='a',valid=True,segment=None):
        return engine.update(video,t,{'angle_deg':value,'angle_valid':int(valid)},segment)

    def test_known_lag_trailing_mean_and_lag_of_mean(self):
        engine=self.engine()
        self.update(engine,0,0)
        first=self.update(engine,.1,10)
        self.assertEqual(first['angle_lag_0p1s_deg'],0)
        self.assertEqual(first['angle_mean_0p2s_valid'],0)
        second=self.update(engine,.2,20)
        self.assertAlmostEqual(second['angle_mean_0p2s_deg'],10)
        self.assertEqual(second['angle_mean_0p2s_lag_0p1s_valid'],0)
        third=self.update(engine,.3,30)
        self.assertAlmostEqual(third['angle_lag_0p1s_deg'],20)
        self.assertAlmostEqual(third['angle_mean_0p2s_deg'],20)
        self.assertAlmostEqual(third['angle_mean_0p2s_lag_0p1s_deg'],10)

    def test_no_history_crosses_videos_or_segments(self):
        engine=self.engine()
        self.update(engine,0,12)
        self.update(engine,.1,20)
        fresh=self.update(engine,.2,30,video='b')
        self.assertEqual(fresh['angle_lag_0p1s_valid'],0)
        fresh=self.update(engine,.2,30,segment='new')
        self.assertEqual(fresh['angle_lag_0p1s_valid'],0)

    def test_missing_angle_and_long_frame_gap_reset_history(self):
        engine=self.engine()
        self.update(engine,0,10)
        self.update(engine,.1,'',valid=False)
        recovered=self.update(engine,.2,20)
        self.assertEqual(recovered['angle_lag_0p1s_valid'],0)
        fresh=self.update(engine,1,30)
        self.assertEqual(fresh['angle_lag_0p1s_valid'],0)
        self.assertEqual(fresh['angle_mean_0p2s_valid'],0)

    def test_lag_never_uses_a_point_after_its_target(self):
        engine=self.engine(lag_tolerance_seconds=.01)
        self.update(engine,0,10)
        self.update(engine,.08,100)
        result=self.update(engine,.15,200)
        # Target is .05s: .08s is closer but is later than the lag target.
        # 0s is outside tolerance, so the value must be missing.
        self.assertEqual(result['angle_lag_0p1s_valid'],0)

    def test_future_values_cannot_change_previous_outputs(self):
        engine=self.engine()
        self.update(engine,0,1)
        self.update(engine,.1,2)
        previous=self.update(engine,.2,3)
        copy=dict(previous)
        self.update(engine,.3,10000)
        self.assertEqual(previous,copy)
        self.assertEqual(previous['angle_mean_0p2s_deg'],2)

    def test_invalid_configuration_and_time_order_rejected(self):
        with self.assertRaises(ValueError):self.engine(lag_seconds=(0,))
        with self.assertRaises(ValueError):self.engine(mean_lag_seconds=float('nan'))
        engine=self.engine();self.update(engine,0)
        with self.assertRaises(ValueError):self.update(engine,0)

    def test_velocity_uses_actual_time_and_retains_direction(self):
        engine=self.engine()
        first=self.update(engine,0,0)
        self.assertEqual(first['angle_velocity_valid'],0)
        rising=self.update(engine,.1,10)
        self.assertAlmostEqual(rising['angle_velocity_deg_per_s'],100)
        falling=self.update(engine,.15,5)
        self.assertAlmostEqual(falling['angle_velocity_deg_per_s'],-100)
        unchanged=self.update(engine,.2,5)
        self.assertEqual(unchanged['angle_velocity_valid'],1)
        self.assertEqual(unchanged['angle_velocity_deg_per_s'],0)

    def test_velocity_does_not_cross_missing_values_or_boundaries(self):
        engine=self.engine()
        self.update(engine,0,10)
        invalid=self.update(engine,.1,'',valid=False)
        self.assertEqual(invalid['angle_velocity_valid'],0)
        recovered=self.update(engine,.2,30)
        self.assertEqual(recovered['angle_velocity_valid'],0)
        gap=self.update(engine,1,40)
        self.assertEqual(gap['angle_velocity_valid'],0)
        new_segment=self.update(engine,1.1,50,segment='new')
        self.assertEqual(new_segment['angle_velocity_valid'],0)
        new_video=self.update(engine,0,50,video='b')
        self.assertEqual(new_video['angle_velocity_valid'],0)


if __name__=='__main__':unittest.main()
