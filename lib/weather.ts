import axios from "axios";

export async function fetchWeather(city: string) {
  const key = process.env.NEXT_PUBLIC_WEATHER_API_KEY;
  const { data } = await axios.get(
    `https://api.weatherapi.com/v1/current.json?key=${key}&q=${city}`
  );
  return data;
}
