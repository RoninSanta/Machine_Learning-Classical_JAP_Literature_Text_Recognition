# Machine Learning - Ancient *Kuzushiji* Text Recognition
***Kuzushiji***, a ancient form of cursive writing that is used for centuries prior to 19th century Japan. Majority of historical documents that exist in the archives of Japan are written in this form of writing. Therefore, it would be great to have a model that could transcribe ancient Kuzushiji scripts into modern characters for researchers to understand what is recorded.


## [1. Background]
What was the weather like 500 years ago? What happened when Mt. Fuji erupted? What subjects did the schools teach back then? Historical documents gives us a window into the past, we are able to glimpse at the world before our time; admire its culture, norms and values reflected against our own. The history of Japan is long, unique, and filled with as many conflicts as there were romances, some remained to be undeciphered till this day. Historically, Japan was isolated from the West it was until the Meiji restoration in 1868, where Japan opened its borders and start its modernization. Due to this drastic modernization, changes have to be made to Japanese language, writing and printing systems in hope to create a standardize system that could easily interpreted by anyone and keep up with the industrial age of the Western world. This led to the `Kuzushiji` script slowly being forgotten, despite the fact that it has been practiced over a 1000 years.

![ancient](https://github.com/RoninSanta/Machine_Learning-Classical_JAP_Literature_Text_Recognition/assets/109457795/86171251-6ded-4340-8b5c-5caae431d682)
Figure 1. A book written in **Kuzushiji** during the Edo period,"Onna Daigaku" published in 1772

![ML JAP Kuzushiji Text Recognition PDF](https://github.com/RoninSanta/Machine_Learning-Classical_JAP_Literature_Text_Recognition/assets/109457795/02556fcb-f6f8-4e88-a2e8-ecc593934c5d)

Figure 2. A textbook taught in universities in attempt to standardize modern Japanese language alphabet ***Hiragana*** , "Shinpen Shushinkyouten Vol.3" published in 1900


## [2. Kuzushiji Dataset]
I will be using a TensorFlow dataset called Kuzushiji-MNIST, a dropin replacement for the MNIST dataset(28x28 grayscale, 70,000 images), this is only a small portion out of a ginormous project that plans to digitize about 300,000 old Japanese records. From this dataset, I will create a model that could recognise the
Kuzushiji texts from the image provided and predict the modern Japanese script it belongs to. Since MNIST restricts to 10 classes, I have to choose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST. For example, shown in the Figure below, there are different ways that a character can be represented with slight variation in handwriting.

<img src="https://github.com/RoninSanta/Machine_Learning-Classical_JAP_Literature_Text_Recognition/assets/109457795/bea74f50-91d6-4bfa-90a7-aeb02ece7811" width="250" height="250">
<img src="https://github.com/RoninSanta/Machine_Learning-Classical_JAP_Literature_Text_Recognition/assets/109457795/9021d2a1-5690-46b0-b316-fb01e0033dc8" width="250" height="250">

Figure 3: The 10 primary classes of Kuzushiji-MNIST dataset.

#### [2.1 Split, test and train sets]
As we would be training the model with training and validation sets, we would need to further split the dataset into training, validation and test sets in a ration of 3:1:1. The output below shows the split of the kmnist dataset between images and labels.
kmnist_train_img: (42000, 28, 28)
kmnist_train_labels: (42000,)
kmnist_val_img: (14000, 28, 28)
kmnist_val_labels: (14000,)
kmnist_test_img: (14000, 28, 28)
kmnist_test_labels: (14000,)

## [3. First Baseline]
Before preparing the data, we will be generating our first baseline based on informed guesswork with the data available to us currently. Our baseline data model will predict solely according to the most populated label/class on the dataset, which is 9(wo) and using the label against the test data to get a probability of accurate prediction.

#### [3.1 One-Hot Encoding]
Next, the model would require categorically encoded labels where each label will be turned into a 10 element vector with a single 'hot' nonzero entry. This is referred as one-hot encoding. Where the position of the vector corresponds to the label will be encoded.

```
# Encode with the convenient to_catergorical function
from tensorflow.keras.utils import to_categorical
#origin_label = test_labels[0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)
```

#### [3.2 Optimizer]
I will be using ADAM a variant of Stochastic Gradient Descent algorithm as our optimizer for this project to reduce loss over many training loops and improve the network prediction. Adam  works very well in training deep learning models and actually outperforms other Adaptive techniques.

#### [3.3 Accuracy VS Loss]
We will be using accuracy as our primary metric at compilation, which is useful since we have same amount of sample per label(refer to first baseline). This is because accuracy is affected by the number of samples available.


## [4. Second Baseline]
Now we will determine what is the baseline without referencing the dataset but simply looking at the labels/classes present in it.

#### [4.1 Underfit, Overfit and Regularizing the Model]
To create the optimal model we need to perform underfitting and overfitting and lastly. After creating a model that overfits, we need to regularize the model as to discourage the complexity of the model and avoid overfitting in the future so as to achieve a perfect balance for the model.

#### [4.2 Final Model]
Hopefully, we have created a model of good-fit, now we will perform training on the complete set instead of the split set. The `confusion matrix` below should be a good indicator of the performance of the model(Predicted vs Truth)


<img src="https://github.com/RoninSanta/Machine_Learning-Classical_JAP_Literature_Text_Recognition/assets/109457795/07360837-087f-4a34-a14a-61b4992b8d6a" width="300" height="300">

## [5. Iterative K-Fold]
Normally, the most common method used is Hold-Out validation however this method is highly dependent on data points within training and testing sets, which in turn is highly reliant on the splitting of dataset into training and test sets.

- Our KMNITS dataset only contains 70,000 samples, the small sample size makes it hard to split, since it will result in small validation points and partial training sets.
- The small validation points makes it highly sensitive to data point changes on split data.
- K-fold validation would be a better choice for evaluating protocols as it reduces the sensitivity of small sets.
- As a result, iterated K-Fold validation would be my preferred evaluation protocol
