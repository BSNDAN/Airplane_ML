import pre_process
import constants
import data_loader
import model 

from tensorflow import keras

#Import data
choosen_output, partition, labels, len_output = pre_process.specifiy_problem()

# Choose data loader parameters
params = {'batch_size': constants.BATCH_SIZE,
          'n_classes': len_output,
          'n_channels':constants.N_CHANNELS,
          'shuffle': True,
         }


training_generator = data_loader.DataGenerator(partition['train'], labels, **params)
validation_generator = data_loader.DataGenerator(partition['val'], labels, **params)

#Build model
models = model.build_Unet(dim=(constants.IMAGE_WIDTH,constants.IMAGE_HEIGHT,constants.IMAGE_DEPTH),
                          nb_output=len_output, activation="softmax",
                         )

opt = keras.optimizers.Nadam(learning_rate=0.001)

models.compile(loss='sparse_categorical_crossentropy', 
               optimizer=opt,
               metrics = ["accuracy"],
              )

#Train model
history = models.fit(training_generator,
                     validation_data = validation_generator,
                     epochs=10,
                     verbose = 1,
                    )

#Save model
models.save(f"../saved_models/{choosen_output}.h5")