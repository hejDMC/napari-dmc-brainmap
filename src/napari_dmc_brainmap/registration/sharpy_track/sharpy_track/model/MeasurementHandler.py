import json

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
        #TODO: load measurement record from json
        # if self.regViewer.status.currentSliceNumber in self.
        return

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








