const tf = require('@tensorflow/tfjs-node');
const tic_tac_toeTesting = require("./data/tic-tac-toe-testing-simulating-game.json")

const testingData = tf.tensor2d(tic_tac_toeTesting.map(item => [
  item.LU, item.UU, item.RU, item.CL, item.CC, item.CR, item.LB, item.BB, item.RB
]))

async function run() {

  const modelUrl = `file://${__dirname}/models/model.json`;
  const model = await tf.loadLayersModel(modelUrl);
  model.summary();


  model.predict(testingData).print();
}

run().then(() => {
  console.log('Done')
})
