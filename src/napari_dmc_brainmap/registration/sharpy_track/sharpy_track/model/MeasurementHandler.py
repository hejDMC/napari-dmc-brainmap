import json
import numpy as np
from src.napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from src.napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.TreRow import TreRow

class MeasurementHandler:
    def __init__(self, measurementPage) -> None:
        self.regViewer = measurementPage.regViewer
        self.load_json()
    
    def load_json(self):
        # check if measurement.json exist
        json_path = self.regViewer.status.folderPath.joinpath('measurement.json')
        if json_path.is_file():
            with open(json_path, 'r') as json_file:
                self.json_data = json.load(json_file)
        else:
            self.create_measurement_template()
    
    def create_measurement_template(self):
        # create measurement.json template
        self.json_data = dict(sourceDots=dict(),
                              useTransformation=dict(),
                              targetDots=dict(),
                              truthDots=dict(),
                              treScore=dict(),
                              imgName=dict())
        # save to measurement.json
        with open(self.regViewer.status.folderPath.joinpath('measurement.json'), 'w') as json_file:
            json.dump(self.json_data, json_file)

    
    def load_measurement_record(self):
        # check if current sample has measurement record
        if str(self.regViewer.status.currentSliceNumber) in self.json_data["imgName"]:
            self.paint_rows()
        else:
            pass

    def paint_rows(self):
        imgIndex = self.regViewer.status.currentSliceNumber
        # set imgIndex
        self.regViewer.measurementPage.active_rows["imgIndex"] = imgIndex
        for source_xy in self.json_data["sourceDots"][imgIndex]:
            self.regViewer.measurementPage.active_rows["source_coords"].append(source_xy)
            # recalculate tranformation matrix from registration json
            tform = fitGeoTrans(self.regViewer.status.sampleDots[imgIndex], 
                                self.regViewer.status.atlasDots[imgIndex])
            self.regViewer.measurementPage.active_rows["tform_matrix"] = tform.tolist()
            # remap target coordinates
            target_xy = mapPointTransform(source_xy[0], source_xy[1], tform)
            target_xy = [np.round(target_xy[0]).astype(int), 
                         np.round(target_xy[1]).astype(int)]
            self.regViewer.measurementPage.active_rows["target_coords"].append(target_xy)

        for truth_xy in self.json_data["truthDots"][imgIndex]:
            self.regViewer.measurementPage.active_rows["truth_coords"].append(truth_xy)
        # calculate tre score
        for tru,tar in zip(self.regViewer.measurementPage.active_rows["truth_coords"],
                           self.regViewer.measurementPage.active_rows["target_coords"]):
            TRE = np.sqrt((tru[0] - tar[0])**2 + (tru[1] - tar[1])**2)
            self.regViewer.measurementPage.active_rows["tre_score"].append(np.round(TRE, 4).astype(str))
        # create row object (remove_btn disabled)
        for row_i in range(len(self.regViewer.measurementPage.active_rows["tre_score"])):
            row_obj = TreRow(self.regViewer.measurementPage)
            row_obj.source_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['source_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['source_coords'][row_i][1]})")
            row_obj.target_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['target_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['target_coords'][row_i][1]})")
            row_obj.true_pos_label.setText(f"({self.regViewer.measurementPage.active_rows['truth_coords'][row_i][0]}, {self.regViewer.measurementPage.active_rows['truth_coords'][row_i][1]})")
            row_obj.tre_score_label.setText(f"{float(self.regViewer.measurementPage.active_rows["tre_score"][row_i]):.2f}")
        # create dot objects
        self.regViewer.widget.viewerRight.addSourceDot()
        self.regViewer.widget.viewerLeft.addTruthDot()

        # enable remove_btn



    def save_measurement_record(self):
        # check if current sample has measurement record
        if len(self.regViewer.measurementPage.active_rows["source_coords"]) == 0:
            return
        else: # save record and clear page
            self.json_data["sourceDots"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.measurementPage.active_rows["source_coords"]
            self.json_data["useTransformation"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.measurementPage.active_rows["tform_matrix"]
            self.json_data["targetDots"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.measurementPage.active_rows["target_coords"]
            self.json_data["truthDots"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.measurementPage.active_rows["truth_coords"]
            self.json_data["treScore"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.measurementPage.active_rows["tre_score"]
            self.json_data["imgName"][self.regViewer.measurementPage.active_rows["imgIndex"]] = self.regViewer.status.imgFileName[self.regViewer.measurementPage.active_rows["imgIndex"]]
            # update measurement.json
            with open(self.regViewer.status.folderPath.joinpath('measurement.json'), 'w') as json_file:
                json.dump(self.json_data, json_file)
            # cleanup page
            for _ in range(len(self.regViewer.measurementPage.active_rows["row_obj"])):
                self.regViewer.measurementPage.active_rows["row_obj"][0].remove_registered_row()

            self.regViewer.measurementPage.active_rows["tform_matrix"] = None
            self.regViewer.measurementPage.active_rows["imgIndex"] = None








