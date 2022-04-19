import axios from "axios";

const baseUrl = "https://t06twtw4n1.execute-api.us-west-1.amazonaws.com/dev/transform";

function transform(data) {
  return axios.post(baseUrl, data);
}

export { transform };
