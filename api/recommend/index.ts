// index.ts  
import { pingInference, returnData, inference } from './helper';

// Incoming lambda request
export async function handler(event: any, _: any) {

  // POST method to submit the feedback
  if (event.httpMethod == "POST") {
    const params = JSON.parse(event.body);
    if (params.userId) {
      return inference(params.userId)
    }
  } else if (event.httpMethod == "GET") {
    // Ping the inference
    return pingInference()
  }

  // Something else that should not happen.
  return returnData({
    message: "Nothing to say"
  })
}  