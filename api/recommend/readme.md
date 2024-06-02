# Recommend.parf.ai - Recommend API

## Introduction

The recommend API takes an image encoded in base64 as argument and returns the recommend score, the classification and the base64 image including the heatmap.

This API is executed on AWS Lambda in the `NodeJS20.x` runtime.
It calls the Sagemaker serverless inference endpoint.

## API Usage example

- Prod: https://recommend.parf.ai/api
- Dev: https://dev.recommend.parf.ai/api

### Recommend an image

```js
const request = require("request");
let options = {
  method: "POST",
  url: "https://recommend.parf.ai/api/recommend",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({"userId":15587}),
};
request(options, (error, response) => {
  if (error) throw new Error(error);
  console.log(response.body);
});
```

Response example

```json
{
    "predictions": {
        "clicked_articles": [
            {
                "article_id": 336221,
                "category_id": 437,
                "score": 0.9999988675117493
            },
            {
                "article_id": 234698,
                "category_id": 375,
                "score": 0.9998106956481934
            },
            {
                "article_id": 161586,
                "category_id": 281,
                "score": 0.9921646118164062
            },
            {
                "article_id": 233605,
                "category_id": 375,
                "score": 0.9868424534797668
            }
        ],
        "top_recommendation": [
            {
                "article_id": 236338,
                "category_id": 375,
                "score": 0.9998152852058411
            },
            {
                "article_id": 336223,
                "category_id": 437,
                "score": 0.9998084902763367
            },
            {
                "article_id": 236552,
                "category_id": 375,
                "score": 0.9996036887168884
            },
            {
                "article_id": 160974,
                "category_id": 281,
                "score": 0.9995703101158142
            },
            {
                "article_id": 160132,
                "category_id": 281,
                "score": 0.999555766582489
            }
        ]
    }
}
```

## How to test locally

```bash
# Go within the recommend api
cd api/recommend

# Install dependencies (using pnpm or npm, yarn, ...)
pnpm install

# Authentication to allow using DB queries and email sending
export DEV_ACCOUNT_ID=REPLACE_ME
export PROD_ACCOUNT_ID=REPLACE_ME
export INFRA_ACCOUNT_ID=REPLACE_ME
export ACCOUNT_ID=$DEV_ACCOUNT_ID

# Login from infra account to dev account
eval $(aws sts assume-role --profile $INFRA_ACCOUNT_ID --role-arn "arn:aws:iam::"$ACCOUNT_ID":role/provision" --role-session-name AWSCLI-Session | jq -r '.Credentials | "export AWS_ACCESS_KEY_ID=\(.AccessKeyId)\nexport AWS_SECRET_ACCESS_KEY=\(.SecretAccessKey)\nexport AWS_SESSION_TOKEN=\(.SessionToken)\n"')

# Run test
npm run test
```

If you need to build the lambda layer for the infrastructure and push it manually
```sh
# Go within the recommend api
cd api/recommend

# Create the lambda layer ready to be used by terraform
npm run prepare:layer
```
