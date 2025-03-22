import { Bar } from "react-chartjs-2";
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

// Register chart components
Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const FraudStats = ({ data = [] }) => {
  // Ensure 'data' is an array before mapping
  if (!Array.isArray(data) || data.length === 0) {
    return <p className="text-center text-gray-400">No data available</p>;
  }

  const chartData = {
    labels: data.map((item) => item.label || "Unknown"),
    datasets: [
      {
        label: "Predicted Fraud",
        data: data.map((item) => item.value || 0),
        backgroundColor: "rgba(59, 130, 246, 0.7)", // blue color for predicted
      },
      {
        label: "Reported Fraud",
        data: data.map((item) => item.reportedValue || 0),
        backgroundColor: "rgba(244, 63, 94, 0.7)", // rose color for reported
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "rgba(229, 231, 235, 0.9)", // light text for dark theme
        },
      },
      tooltip: {
        backgroundColor: "rgba(31, 41, 55, 0.9)", // dark background for tooltip
        titleColor: "rgba(255, 255, 255, 0.9)",
        bodyColor: "rgba(255, 255, 255, 0.9)",
        borderColor: "rgba(75, 85, 99, 0.3)",
        borderWidth: 1,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: "rgba(75, 85, 99, 0.2)", // subtle grid lines
        },
        ticks: {
          color: "rgba(229, 231, 235, 0.7)", // light text for y-axis
        },
      },
      x: {
        grid: {
          display: false, // hide x grid lines
        },
        ticks: {
          color: "rgba(229, 231, 235, 0.7)", // light text for x-axis
        },
      },
    },
  };

  return (
    <div className="h-64 relative">
      <Bar data={chartData} options={options} />
    </div>
  );
};

export default FraudStats;
