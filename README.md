# DOOM Eternal & Animal Crossing: New Horizons Meme & Image Classification
## Inspiration
Remember 2020? When everyone was stuck at home in the middle of pandemic and then two big game titles DOOM and Animal Crossing released their games in the same day and somehow both communities got together rejoiced? It was awesome. \
To celebrate that important event of internet history - or dare I say, **the world's history** - *and for the term project of our Deep Learning class* we trained a model to predict Doom Eternal and Animal Crossing memes of that time.
<p align="center">
  <img src= "https://github.com/beyza1tozman/doom_crossing_project/blob/main/gifs/DCEH%20-%20rip%20and%20tear.gif" alt="rip and tear until it's done" width=480 height=320 /> 
</p>


## Dataset and Technologies
[Doom or Animal Crossing?](https://www.kaggle.com/datasets/andrewmvd/doom-crossing) dataset from Kaggle is used for training. \
A lot of messing about is done with PyTorch and TensorFlow and custom CNN models, though they were poor in performance. Then TensorFlow and MobileNetV2 is used for transfer learning and fine-tuning and it achieved the best results.
## Results 
Our model achieved 90% test accuracy after fine tuning, which appears to be satisfactory for this dataset. \
We also put it to test by predicting the [DOOM CROSSING: Eternal Horizons](https://www.youtube.com/watch?v=U4lz8MN6MQA) music video by The Chalkeaters. ]
<p align="center" >
  <img src= "https://github.com/beyza1tozman/doom_crossing_project/blob/main/gifs/DCEH%20-%20friendship.gif" alt="the light of friendship shines through two dimensions" width=480 height=320 /> 
</p>
<p align="center" >
  <img src= "https://github.com/beyza1tozman/doom_crossing_project/blob/main/gifs/DCEH%20-%20we%20are%20one.gif" alt="we are one" width=480 height=320  /> 
</p>

## Learning Curves
The following plots show the training and validation accuracy, loss of MobileNetV2 during feature extraction and fine-tuning.

<p align="center" >
<img src="https://github.com/beyza1tozman/doom_crossing_project/assets/104080203/be88d179-860c-4706-9688-793d3c87d1dd" alt="MobileNetV2 Feature Extraction - Training and Validation Accuracy" width="350">
<img src="https://github.com/beyza1tozman/doom_crossing_project/assets/104080203/1184409d-c7fd-46b9-b92b-b25085c35554" alt="MobileNetV2 Fine Tuning - Training and Validation Accuracy" width="350">
</p>

Final Test Accuracy: 0.8950892686843872


