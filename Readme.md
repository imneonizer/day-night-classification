## Binary Image classification

Train a model to classify Day / Night images using keras.

Read full article here https://medium.com/@mneonizer/day-night-classification-a01a7d9af695

**Training**

````
python train_model.py --dataset images --model model/day_night.hd5
````

Images directory contains 2 sub-directories

````
├── images
│   ├── day
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   ├── 00003.jpg
│   │   ├── xxxxx.jpg
│   ├── night
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   ├── 00003.jpg
│   │   ├── xxxxx.jpg
````

And the model is trained to classify these two labels: ``day / night``.

**Testing**

````
python test_model.py --model model/day_night.hd5 --image test/n2.jpg
````

![](Docs/r1.jpg)

````
python test_model.py --model model/day_night.hd5 --image test/d1.jpg
````

![](Docs/r2.jpg)