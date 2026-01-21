import React, { useState } from 'react';
import { View, Text, Image, Button, ActivityIndicator, StyleSheet } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Permission required!');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      classify(result.assets[0].uri);
    }
  };

  const classify = async (uri) => {
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', {
        uri,
        name: 'leaf.jpg',
        type: 'image/jpeg',
      });

      const response = await fetch('http://YOUR_SERVER_IP:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResult(`Prediction: ${data.label}\nConfidence: ${data.confidence}%`);
    } catch (error) {
      setResult('Error connecting to server');
    }
    setLoading(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>☕ Coffee Leaf Disease Detector</Text>
      <Button title="Select Image" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={styles.image} />}
      {loading ? <ActivityIndicator size="large" /> : <Text style={styles.result}>{result}</Text>}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  title: { fontSize: 20, fontWeight: 'bold', marginBottom: 20 },
  image: { width: 300, height: 300, marginVertical: 20 },
  result: { fontSize: 16, textAlign: 'center' },
});
