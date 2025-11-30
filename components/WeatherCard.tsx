import {
  FaTemperatureHigh,
  FaTint,
  FaCloudRain,
  FaWind,
} from "react-icons/fa";

interface Props {
  temperature: number;
  humidity: number;
  rainfall: number;
  windspeed: number;
}

export default function WeatherCard({
  temperature,
  humidity,
  rainfall,
  windspeed,
}: Props) {
  const stats = [
    {
      icon: FaTemperatureHigh,
      label: "Temperature",
      unit: "°C",
      value: temperature,
      color: "bg-red-100 text-red-600",
    },
    {
      icon: FaTint,
      label: "Humidity",
      unit: "%",
      value: humidity,
      color: "bg-blue-100 text-blue-600",
    },
    {
      icon: FaCloudRain,
      label: "Rainfall",
      unit: "mm",
      value: rainfall,
      color: "bg-teal-100 text-teal-600",
    },
    {
      icon: FaWind,
      label: "Wind Speed",
      unit: "km/h",
      value: windspeed,
      color: "bg-purple-100 text-purple-600",
    },
  ];

  return (
    
    <div className="bg-white rounded-2xl shadow-lg p-6 grid grid-cols-2 gap-6 hover:shadow-xl transition max-w-md mx-auto">
      {stats.map((stat) => {
        const Icon = stat.icon;
        return (
          <div
            key={stat.label}
            className="flex flex-col items-center justify-center text-center"
          >
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 ${stat.color}`}
            >
              <Icon className="text-xl" />
            </div>
            <span className="text-2xl font-bold text-gray-700">
              {stat.value} {stat.unit}
            </span>
            <span className="text-sm text-gray-500 mt-1">{stat.label}</span>
          </div>
        );
      })}
    </div>
  );
}