import axios from "axios";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export const predictFlood = async (data: FormData) => {
  const res = await axios.post(`${API_URL}/predict/flood`, data, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};

export const predictLandslide = async (data: FormData) => {
  const res = await axios.post(`${API_URL}/predict/landslide`, data, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
};
