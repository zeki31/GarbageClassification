
Main idea:

- Data preparation:

  - Images are loaded from specified directories and preprocessed.
  - The dataset is split into training and testing sets.


- Model architecture:

  - Uses a pre-trained VGG16 model as a feature extractor.
  - Adds a custom classifier on top of the VGG16 features.
  - Implements adaptive pooling to handle different input sizes.


- Training:

  - Uses CrossEntropyLoss as the loss function.
  - Uses SGD optimizer with momentum.
  - Trains for 150 epochs.


- Evaluation:

  - Calculates and prints the test accuracy.


- Utility functions:

  - Includes a function to classify a single image.


- Model saving and loading:

  - Saves the trained model and optimizer state.
  - Includes commented-out code for loading a saved model.


- Test on a single image:

  - Loads and classifies a test image, displaying it and printing the predicted category.



Result for sigle layer label structure:

Epoch [1/150], Loss: 1.8023
Epoch [2/150], Loss: 1.6970
Epoch [3/150], Loss: 1.6357
Epoch [4/150], Loss: 1.7197
Epoch [5/150], Loss: 1.6653
Epoch [6/150], Loss: 1.5784
Epoch [7/150], Loss: 1.5069
Epoch [8/150], Loss: 1.6625
Epoch [9/150], Loss: 1.6590
Epoch [10/150], Loss: 1.7063
Epoch [11/150], Loss: 1.5016
Epoch [12/150], Loss: 1.6807
Epoch [13/150], Loss: 1.5982
Epoch [14/150], Loss: 1.4395
Epoch [15/150], Loss: 1.3537
Epoch [16/150], Loss: 1.4543
Epoch [17/150], Loss: 1.3356
Epoch [18/150], Loss: 1.3589
Epoch [19/150], Loss: 1.4348
Epoch [20/150], Loss: 1.4082
Epoch [21/150], Loss: 1.0366
Epoch [22/150], Loss: 1.5712
Epoch [23/150], Loss: 1.4880
Epoch [24/150], Loss: 1.4469
Epoch [25/150], Loss: 1.2911
Epoch [26/150], Loss: 1.3663
Epoch [27/150], Loss: 1.1939
Epoch [28/150], Loss: 1.2298
Epoch [29/150], Loss: 1.0051
Epoch [30/150], Loss: 1.2570
Epoch [31/150], Loss: 1.3618
Epoch [32/150], Loss: 1.2901
Epoch [33/150], Loss: 1.2509
Epoch [34/150], Loss: 1.1701
Epoch [35/150], Loss: 1.5046
Epoch [36/150], Loss: 1.2165
Epoch [37/150], Loss: 1.4818
Epoch [38/150], Loss: 1.1975
Epoch [39/150], Loss: 1.2927
Epoch [40/150], Loss: 1.3973
Epoch [41/150], Loss: 1.1017
Epoch [42/150], Loss: 0.9736
Epoch [43/150], Loss: 1.2281
Epoch [44/150], Loss: 1.1166
Epoch [45/150], Loss: 1.2402
Epoch [46/150], Loss: 1.1506
Epoch [47/150], Loss: 1.2461
Epoch [48/150], Loss: 1.1929
Epoch [49/150], Loss: 1.1309
Epoch [50/150], Loss: 1.1697
Epoch [51/150], Loss: 1.0690
Epoch [52/150], Loss: 1.1221
Epoch [53/150], Loss: 0.8200
Epoch [54/150], Loss: 1.2761
Epoch [55/150], Loss: 1.1461
Epoch [56/150], Loss: 1.0235
Epoch [57/150], Loss: 1.0517
Epoch [58/150], Loss: 0.9787
Epoch [59/150], Loss: 0.9763
Epoch [60/150], Loss: 1.3128
Epoch [61/150], Loss: 0.9090
Epoch [62/150], Loss: 0.9311
Epoch [63/150], Loss: 0.9356
Epoch [64/150], Loss: 1.1148
Epoch [65/150], Loss: 1.1151
Epoch [66/150], Loss: 0.8583
Epoch [67/150], Loss: 0.9771
Epoch [68/150], Loss: 1.1110
Epoch [69/150], Loss: 0.7720
Epoch [70/150], Loss: 1.0919
Epoch [71/150], Loss: 0.8046
Epoch [72/150], Loss: 1.2803
Epoch [73/150], Loss: 0.9988
Epoch [74/150], Loss: 1.2870
Epoch [75/150], Loss: 0.9489
Epoch [76/150], Loss: 0.9767
Epoch [77/150], Loss: 0.8367
Epoch [78/150], Loss: 1.1513
Epoch [79/150], Loss: 0.9428
Epoch [80/150], Loss: 0.9283
Epoch [81/150], Loss: 1.0074
Epoch [82/150], Loss: 0.8880
Epoch [83/150], Loss: 1.0361
Epoch [84/150], Loss: 1.1107
Epoch [85/150], Loss: 0.8447
Epoch [86/150], Loss: 0.7958
Epoch [87/150], Loss: 1.1410
Epoch [88/150], Loss: 1.2503
Epoch [89/150], Loss: 1.0352
Epoch [90/150], Loss: 0.8567
Epoch [91/150], Loss: 1.2334
Epoch [92/150], Loss: 1.1639
Epoch [93/150], Loss: 0.9495
Epoch [94/150], Loss: 1.0320
Epoch [95/150], Loss: 1.0856
Epoch [96/150], Loss: 0.7583
Epoch [97/150], Loss: 1.3716
Epoch [98/150], Loss: 0.8588
Epoch [99/150], Loss: 0.9637
Epoch [100/150], Loss: 0.8541
Epoch [101/150], Loss: 0.8258
Epoch [102/150], Loss: 0.7131
Epoch [103/150], Loss: 0.7590
Epoch [104/150], Loss: 0.7422
Epoch [105/150], Loss: 1.0452
Epoch [106/150], Loss: 0.8422
Epoch [107/150], Loss: 0.9254
Epoch [108/150], Loss: 1.0221
Epoch [109/150], Loss: 1.1190
Epoch [110/150], Loss: 0.8681
Epoch [111/150], Loss: 0.9491
Epoch [112/150], Loss: 0.8120
Epoch [113/150], Loss: 1.1240
Epoch [114/150], Loss: 0.6322
Epoch [115/150], Loss: 1.0272
Epoch [116/150], Loss: 0.5357
Epoch [117/150], Loss: 0.8919
Epoch [118/150], Loss: 0.5179
Epoch [119/150], Loss: 1.1684
Epoch [120/150], Loss: 0.9739
Epoch [121/150], Loss: 0.9004
Epoch [122/150], Loss: 0.7336
Epoch [123/150], Loss: 0.6871
Epoch [124/150], Loss: 0.6293
Epoch [125/150], Loss: 0.6787
Epoch [126/150], Loss: 0.7063
Epoch [127/150], Loss: 1.1356
Epoch [128/150], Loss: 0.8436
Epoch [129/150], Loss: 0.5994
Epoch [130/150], Loss: 0.7701
Epoch [131/150], Loss: 0.7775
Epoch [132/150], Loss: 1.0532
Epoch [133/150], Loss: 0.7125
Epoch [134/150], Loss: 0.5227
Epoch [135/150], Loss: 0.5352
Epoch [136/150], Loss: 1.0153
Epoch [137/150], Loss: 0.5911
Epoch [138/150], Loss: 0.3856
Epoch [139/150], Loss: 0.5537
Epoch [140/150], Loss: 0.8786
Epoch [141/150], Loss: 0.6763
Epoch [142/150], Loss: 1.0560
Epoch [143/150], Loss: 0.6917
Epoch [144/150], Loss: 0.9018
Epoch [145/150], Loss: 0.6228
Epoch [146/150], Loss: 0.6208
Epoch [147/150], Loss: 0.9390
Epoch [148/150], Loss: 0.7220
Epoch [149/150], Loss: 0.5657
Epoch [150/150], Loss: 0.9721
Test Accuracy: 77.18%
