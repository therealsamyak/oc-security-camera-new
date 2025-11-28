## Optimal Charge Security Camera

We have a battery-powered security camera that runs local pre-trained image-recognition models. A battery is attached both to the charger, and the camera. The camera is context aware about clean energy information externally. Based on the information, we have a controller that looks at current battery percentage, the energy coming in externally, and user-defined metrics (minimum accuracy, minimum latency, how often we run the model etc.), to decide which image-recognition model is loaded based on all these factors. We want to decide when to charge the battery vs when to use the energy for higher accuracy. 

THE GOAL OF THIS PAPER IS TO OBTAIN A MODEL WITH GOOD ENOUGH WEIGHTS SO IT CAN BE RAN GENERAL-USE IN ALL SCENARIOS

### Battery

The battery will be software simulated based on real-world battery behavior. We need a test-benchmark to measure the power usage of each model task for all model sizes, which we can then use to simulate the battery behavior. Apparently, this can be done by measuring the CPU clock or some sort of API, I am not 100% sure.

### Image Models

The image models our controller will pick from are the YOLOv10 series of models, whose specifications are found in the `datasets/model-data.csv` file. 

### Task

The task given to all models is to detect and classify the object in two different images, both found under the `benchmark-images/` directory. The output for both images should be 1 human only.

### Clean Energy Data

The clean energy data is found in the `datasets/` directory, and includes data over the year 2024 in 5-minute intervals. There are 4 CSVs, each corresponding to a region of the USA. 

###