
// const * as tf = require("@tensorflow/tfjs")
const tf  = require("@tensorflow/tfjs-node")
const tic_tac_toe = require("./data/tic-tac-toe.json")
const tic_tac_toeTesting = require("./data/tic-tac-toe-testing.json")

// convert/setup our data
const trainingData = tf.tensor2d(tic_tac_toe.map(item => [
  item.LU, item.UU, item.RU, item.CL, item.CC, item.CR, item.LB, item.BB, item.RB
]))

console.log(tic_tac_toe.filter(item => item.result !== "positive" && item.result !== "negative"))
const outputData = tf.tensor2d(tic_tac_toe.map(item => [
  item.result === "positive" ? 1 : 0,
  item.result === "negative" ? 1 : 0
]))
const testingData = tf.tensor2d(tic_tac_toeTesting.map(item => [
  item.LU, item.UU, item.RU, item.CL, item.CC, item.CR, item.LB, item.BB, item.RB
]))

// build neural network
const model = tf.sequential()

model.add(tf.layers.dense({
  inputShape: [9],
  activation: "sigmoid",
  units: 18,
}))

model.add(tf.layers.dense({
  inputShape: [18],
  activation: "sigmoid",
  units: 9,
}))
model.add(tf.layers.dense({
  activation: "sigmoid",
  units: 2,
}))

model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(.06),
})
// train/fit our network
const startTime = Date.now()
model.fit(trainingData, outputData, {epochs: 10000})
  .then(async (history) => {
    // console.log(history)
    model.predict(testingData).print()
    await model.save(`file://${__dirname}/models`);

  })
// test network