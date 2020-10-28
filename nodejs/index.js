import fs from 'fs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs-node';

const imageCat = `${__dirname}/assets/cat.jpg`;
const imageDog = `${__dirname}/assets/dog.png`;

const image = fs.readFileSync(imageCat);
const decodedImage = tf.node.decodeImage(image, 3);

async function App() {
  const model = await mobilenet.load();
  const predictions = await model.classify(decodedImage);
  console.log(`Predictions: ${JSON.stringify(predictions, undefined, 2)}`);
}

App();
