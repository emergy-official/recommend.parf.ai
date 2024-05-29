import { SageMakerRuntimeClient, InvokeEndpointCommand } from "@aws-sdk/client-sagemaker-runtime";
import fs from "fs"
import { handler } from "./index";
/*
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_REGION=us-east-1
export INFERANCE_NAME=recommend-api
export INFERANCE_URL=https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/recommend-api
*/

const runtimeClient = new SageMakerRuntimeClient({ region: "us-east-1" });

const main = async () => {
    const res = await handler({
        httpMethod: "POST",
        body: JSON.stringify({
            userId: 50
        })
    }, "")
    console.log(res)
}

main()