import numpy as np
from scipy.io import savemat


class Nfb:

    def __init__(self, session, iteration):

        self.session = session
        self.iteration = iteration
        self.block_nf = 0
        self.first_nf = 0
        self.disp_value = 0

        # for NFB
        self.condition = 0
        self.end_psc = False
        self.disp_value = 0
        self.disp_values = np.zeros((self.session.nr_vols, 1))
        self.reward = ""
        self.display_data = {"disp_stage": "",
                             "feedback_type": "",
                             "condition": 0,
                             "disp_value": 0,
                             "reward": "",
                             "iteration": 0,
                             "disp_blank_screen": False,
                             "task_seq": False}

        self.norm_perc_values = []
        self.vect_nfbs = np.zeros((self.session.nr_vols, 1))

    def nfb_init(self):

        if self.iteration.iter_number < self.session.config.skip_vol_nr:
            return

        if self.session.config.type == "PSC" or self.session.config.type == "SVM":
            iter_norm_number = self.iteration.iter_norm_number

            condition = self.session.vect_end_cond[iter_norm_number]
            self.condition = condition

            prot = self.session.config.prot
            if prot == "Cont":

                if condition == 1:
                    self.end_psc = False
                    self.disp_value = 0
                    self.reward = ""
                elif condition == 2:
                    self.end_psc = 1
                elif condition == 3:
                    self.end_psc = 0
                    self.reward = ""
                self.display_data["disp_stage"] = "instruction"
                self.display_data["feedback_type"] = "bar_count"

            elif prot == "ContTask":

                if condition == 1:
                    self.end_psc = 0
                    self.disp_value = 0
                    self.reward = ""
                elif condition == 2:
                    self.end_psc = 1
                elif condition == 3:
                    self.end_psc = 0
                    self.reward = ""
                self.display_data["disp_stage"] = "instruction"
                self.display_data["feedback_type"] = "bar_count_task"

            elif prot == "Inter":

                if condition in [1, 2, 3, 4, 5]:
                    self.end_psc = 0
                    self.disp_value = 0
                    self.reward = ""
                    self.display_data["disp_stage"] = "instruction"
                elif condition == 6:
                    self.end_psc = 1
                    self.display_data["disp_stage"] = "feedback"
                self.display_data["feedback_type"] = "value_fixation"

            self.display_data["condition"] = condition
            self.display_data["disp_value"] = self.disp_value
            self.display_data["reward"] = self.reward

        self.display_data["iteration"] = self.iteration.iter_number
        self.display_data["disp_blank_screen"] = 0
        self.display_data["task_seq"] = 0

    def nfb_calc(self, is_rtqa):

        # Calculate haemodynamic delay (6 s) in volumes
        nVolDelay = np.ceil(6000/self.session.config.tr).astype(int)

        type = self.session.config.type
        condition = self.condition
        ind_vol_norm = self.iteration.iter_norm_number
        if is_rtqa:
            nr_rois = self.session.nr_rois-1
        else:
            nr_rois = self.session.nr_rois
        scale_time_series = self.iteration.mr_time_series.scale_time_series

        max_fb_value = self.session.config.max_feedback_val
        min_fb_value = self.session.config.min_feedback_val
        fb_val_dec = self.session.config.feedback_val_dec

        if type == "PSC" and (self.session.config.prot == "Cont" or self.session.config.prot == "ContTask"):

            block_nf = self.block_nf
            first_nf = self.first_nf

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds[0]:
                    block_nf = np.where(self.session.first_nf_inds[0] == ind_vol_norm)[0][0]
                    first_nf = ind_vol_norm

                i_block_bas = np.array([], dtype=np.int32)
                for i_bas in range(0, block_nf + 1):
                    i_block_bas = np.append(i_block_bas, self.session.prot_cond[0][i_bas][nVolDelay:])

                norm_perc_values = np.zeros((nr_rois,1))
                for i_roi in range(0, nr_rois):
                    m_bas = np.median(scale_time_series[i_roi][i_block_bas])
                    m_cond = scale_time_series[i_roi][ind_vol_norm]
                    norm_perc_values[i_roi] = m_cond - m_bas

                tmp_fb_val = np.median(norm_perc_values)
                self.disp_value = np.round(max_fb_value * tmp_fb_val, decimals=fb_val_dec)

                if not self.session.config.neg_feedback and self.disp_value < 0:
                    self.disp_value = 0
                elif not self.session.config.neg_feedback and self.disp_value < min_fb_value:
                    self.disp_value = min_fb_value
                if self.disp_value > max_fb_value:
                    self.disp_value = max_fb_value

                self.norm_perc_values.append(norm_perc_values)
                self.disp_values[ind_vol_norm] = self.disp_value
            else:

                tmp_fb_val = 0
                self.disp_value = 0
                self.norm_perc_values.append(np.zeros((nr_rois, 1)))

            self.vect_nfbs[ind_vol_norm] = tmp_fb_val
            self.block_nf = block_nf
            self.first_nf = first_nf

            self.display_data["reward"] = ''
            self.display_data["disp_value"] = self.disp_value

        if type == "SVM":

            block_nf = self.block_nf
            first_nf = self.first_nf

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds[0]:
                    block_nf = np.where(self.session.first_nf_inds[0] == ind_vol_norm)[0][0]
                    first_nf = ind_vol_norm

                norm_perc_values = np.zeros((nr_rois, 1))
                for i_roi in range(0, nr_rois):
                    norm_perc_values[i_roi] = scale_time_series[i_roi][ind_vol_norm]

                tmp_fb_val = norm_perc_values.mean()
                tmp_tmp_fbVal = np.round(max_fb_value * tmp_fb_val, decimals=fb_val_dec)
                self.disp_value = tmp_tmp_fbVal

                self.norm_perc_values.append(norm_perc_values)
                self.disp_values[ind_vol_norm] = self.disp_value

            else:

                tmp_fb_val = 0
                self.norm_perc_values.append(np.zeros((nr_rois, 1)))
                self.disp_value = 100

            self.vect_nfbs[ind_vol_norm] = tmp_fb_val
            self.block_nf = block_nf
            self.first_nf = first_nf

            self.display_data["reward"] = ''
            self.display_data["disp_value"] = self.disp_value

        if type == "PSC" and self.session.config.prot == "Inter":
            
            block_nf = self.block_nf
            first_nf = self.first_nf

            if condition == 2:

                if ind_vol_norm in self.session.first_nf_inds[0]:
                    block_nf = np.where(self.session.first_nf_inds[0] == ind_vol_norm)[0][0]
                    first_nf = ind_vol_norm

                if first_nf == ind_vol_norm:
                    i_block_nf = self.session.prot_cond[1][block_nf][nVolDelay:]
                    i_block_bas = self.session.prot_cond[0][block_nf][nVolDelay:]

                    if block_nf >= 1:
                        i_block_bas = np.append(i_block_bas, i_block_bas[-1]+range(1,nVolDelay))







                    # --- PSC Calculation ---
                    # We need dynamic min/max scaling limits. 
                    # Assuming self.iteration.mr_time_series has mpos_min/max arrays matching ind_vol_norm
                    try:
                        mpos_min = self.iteration.mr_time_series.mpos_min[ind_vol_norm]
                        mpos_max = self.iteration.mr_time_series.mpos_max[ind_vol_norm]
                    except AttributeError:
                        # Fallback if properties missing (should be in mrtimeseries.py)
                        mpos_min = 0
                        mpos_max = 1 
                        
                    norm_perc_values = np.zeros((nr_rois, 1))
                    
                    for i_roi in range(nr_rois):
                        # Get data for ROI
                        ts_data = scale_time_series[i_roi]
                        
                        # Averaging (Median)
                        # MATLAB: median(mainLoopData.scalProcTimeSeries(indRoi, i_blockBAS))
                        # Python: slicing with integer array
                        m_bas = np.median(ts_data[i_block_bas])
                        m_cond = np.median(ts_data[i_block_nf])

                        # Scaling: (val - min) / (max - min)
                        range_val = mpos_max - mpos_min
                        if range_val == 0: range_val = 1.0 # Avoid div by zero

                        m_bas_scaled = (m_bas - mpos_min) / range_val
                        m_cond_scaled = (m_cond - mpos_min) / range_val
                        
                        norm_perc_values[i_roi] = m_cond_scaled - m_bas_scaled

                    # Compute average %SC feedback value
                    # MATLAB: eval(P.RoiAnatOperation); Default is usually mean
                    tmp_fb_val = np.mean(norm_perc_values)

                    self.vect_nfbs[ind_vol_norm] = tmp_fb_val
                    disp_value = np.round(max_fb_value * tmp_fb_val, decimals=fb_val_dec)

                    # Clamping [Min...Max]
                    if not self.session.config.neg_feedback and disp_value < 0:
                        disp_value = 0
                    elif self.session.config.neg_feedback and disp_value < min_fb_value:
                        disp_value = min_fb_value
                    
                    if disp_value > max_fb_value:
                        disp_value = max_fb_value

                    # --- RegSuccess and Shaping Logic ---
                    # Initialize act_value storage if not present
                    if not hasattr(self.session, 'act_value'):
                        self.session.act_value = {} # Using dict to map block_index -> value
                    
                    self.session.act_value[block_nf] = tmp_fb_val
                    current_act_val = self.session.act_value[block_nf]

                    nf_run_nr = self.session.config.nf_run_nr # Assuming this exists in config
                    
                    if nf_run_nr == 1:
                        if block_nf == 0: # First block (Python index 0)
                            if current_act_val > 0.5:
                                reg_success = 1
                        else:
                            # Compare with previous blocks
                            if block_nf == 1:
                                tmp_prev = self.session.act_value[0]
                            elif block_nf == 2:
                                vals = [self.session.act_value[0], self.session.act_value[1]]
                                tmp_prev = np.median(vals)
                            else:
                                # Median of last 3 excluding current? 
                                # MATLAB: median(P.actValue(blockNF-3:blockNF-1)) -> indices -3, -2, -1 relative to current
                                vals = [self.session.act_value[block_nf-3], 
                                        self.session.act_value[block_nf-2], 
                                        self.session.act_value[block_nf-1]]
                                tmp_prev = np.median(vals)
                            
                            if (0.9 * current_act_val >= tmp_prev):
                                reg_success = 1

                    elif nf_run_nr > 1:
                        # Logic for subsequent runs using previous run data
                        # Assuming prev_act_value exists in session (list of values from prev run)
                        if hasattr(self.session, 'prev_act_value'):
                            # Create combined vector: [prev_run_values, current_run_values]
                            # Converting dict values to list for current run
                            curr_vals = [self.session.act_value[k] for k in range(block_nf + 1)]
                            tmp_act_value = np.concatenate((self.session.prev_act_value, curr_vals))
                            
                            lact_val = len(tmp_act_value)
                            # Median of last 3, excluding current (indices -4, -3, -2 from end?)
                            # MATLAB: tmp_actValue(lactVal-3:lactVal-1)
                            tmp_prev = np.median(tmp_act_value[lact_val-4 : lact_val-1]) 
                            
                            if (0.9 * current_act_val >= tmp_prev):
                                reg_success = 1

                    self.norm_perc_values.append(norm_perc_values)
                    # self.session.reg_success[block_nf] = reg_success # If storage needed
                    
                    # Update display value only at the calculation point
                    self.disp_values[ind_vol_norm] = disp_value
                    self.disp_value = disp_value # Update class property
                    
                else:
                    tmp_fb_val = 0
            else:
                tmp_fb_val = 0
            
            # Final updates for this volume
            if self.end_psc:
                # Keep showing the calculated value
                self.disp_values[ind_vol_norm] = self.disp_value
            else:
                self.disp_values[ind_vol_norm] = 0
                self.disp_value = 0

            self.vect_nfbs[ind_vol_norm] = tmp_fb_val
            self.block_nf = block_nf
            self.first_nf = first_nf
            self.display_data["reward"] = '' # Set appropriate reward string if logic exists
            self.display_data["disp_value"] = self.disp_value

    def nfb_save(self, save_path):

        if not save_path.is_dir():
            save_path.mkdir(exist_ok=True)

        path = save_path / "py_disp_values.mat"

        if not path.is_file():
            path.touch(exist_ok=True)

        savemat(str(path), {"disp_values": self.disp_values,
                            "vect_end_cond": self.session.vect_end_cond,
                            "norm_perc_values": self.norm_perc_values
                            })
