package com.example.coffeeguard

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var resultText: TextView
    private lateinit var progressBar: ProgressBar
    private val imageSize = 224
    private val numClasses = 4
    private val labels = arrayOf("Healthy", "Leaf Rust", "Miner", "Phoma")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultText = findViewById(R.id.resultText)
        progressBar = findViewById(R.id.progressBar)
        val selectButton = findViewById<Button>(R.id.selectButton)
        val cameraButton = findViewById<Button>(R.id.cameraButton)

        interpreter = loadModelFile()

        // Gallery selection
        val galleryLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                classifyImage(bitmap)
            }
        }

        // Camera capture
        val cameraLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val bitmap = result.data?.extras?.get("data") as? Bitmap
                bitmap?.let { classifyImage(it) }
            }
        }

        selectButton.setOnClickListener { galleryLauncher.launch("image/*") }
        cameraButton.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            cameraLauncher.launch(intent)
        }
    }

    private fun loadModelFile(): Interpreter {
        return try {
            val afd = assets.openFd("coffee_model.tflite")
            val inputStream = FileInputStream(afd.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = afd.startOffset
            val declaredLength = afd.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            Interpreter(modelBuffer, Interpreter.Options())
        } catch (e: IOException) {
            throw RuntimeException("❌ Failed to load model: ${e.message}")
        }
    }

    private fun classifyImage(bitmap: Bitmap) {
        progressBar.visibility = ProgressBar.VISIBLE
        imageView.setImageBitmap(bitmap)

        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, false)
        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
        val output = Array(1) { FloatArray(numClasses) }
        interpreter.run(byteBuffer, output)

        val confidences = output[0]
        val maxIdx = confidences.indices.maxByOrNull { confidences[it] } ?: -1
        val label = labels[maxIdx]
        val confidence = confidences[maxIdx] * 100

        progressBar.visibility = ProgressBar.GONE
        resultText.text = "Prediction: $label\nConfidence: %.2f%%".format(confidence)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(imageSize * imageSize)
        bitmap.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        var pixelIndex = 0
        for (i in 0 until imageSize) {
            for (j in 0 until imageSize) {
                val pixel = pixels[pixelIndex++]
                byteBuffer.putFloat(((pixel shr 16 and 0xFF) / 255.0f))
                byteBuffer.putFloat(((pixel shr 8 and 0xFF) / 255.0f))
                byteBuffer.putFloat(((pixel and 0xFF) / 255.0f))
            }
        }
        return byteBuffer
    }
}
