from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_gen = train_datagen.flow_from_directory(
    r"C:\Users\tluke\Desktop\tensorflowproject\Dataset",
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


print(train_gen.class_indices)


