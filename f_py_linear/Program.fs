open NumSharp
open Tensorflow
open type Binding
open type KerasApi

let tf = New<tensorflow>()
tf.enable_eager_execution()

// Parameters
let traStep = 100
let leaRate = 0.0001f
let disStep = 10

// Sample data
let trX = 
    np.array(12.3999f, 14.3f, 14.5f, 14.8999f, 16.1f, 16.899f, 16.5f, 15.399f, 17.0f, 17.899f,
             18.7999f, 20.2999f, 22.3999f, 19.32f, 15.5f, 16.7f)
let trY = 
    np.array(11.1999f, 12.5f, 12.699f, 13.1f, 14.100f, 14.8f, 14.39f,
             13.3999f, 14.8999f, 15.6f, 16.3999f, 17.700f, 19.60f, 16.89f, 14.0f, 14.6f)
let NShape = trX.shape.[0]

printfn "%i %i" NShape trY.shape.[0]

for i = 0 to NShape - 1 do
    printfn "%A, %A" trX.[i] trY.[i]
let W = tf.Variable(1.03f,name = "weight")
let b = tf.Variable(-0.01f, name = "bias")
let optimizer = keras.optimizers.SGD(leaRate)

// Run training for the given number of steps.
for step = 1 to  (traStep + 1) do 
    use g = tf.GradientTape()
    let pred = W * trX + b
    let loss = tf.reduce_sum(tf.pow(pred - trY,2)) / (2 * NShape)
    let gradients = g.gradient(loss,struct (W,b))

    optimizer.apply_gradients(zip(gradients, struct (W,b)))

    if (step % disStep) = 0 then
        let pred = W * trX + b
        let loss = tf.reduce_sum(tf.pow(pred-trY,2)) / (2 * NShape)
        printfn $"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}"
  
printfn "Тест:"
let prov_ = 12.32f * W.numpy() + b.numpy()      
printfn $"W: {W.numpy()}, b: {b.numpy()} X = 12.32f, Y(pred) = {prov_}"