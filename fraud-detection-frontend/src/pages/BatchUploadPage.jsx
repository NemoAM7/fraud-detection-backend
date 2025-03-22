import BatchUpload from "../components/BatchUpload";

export default function BatchUploadPage() {
  return (
    <div className="p-6 bg-gray-800 rounded-lg shadow-lg">
      <h2 className="text-2xl font-normal text-white mb-4">📂 Batch Upload</h2>
      <BatchUpload />
    </div>
  );
}
