import React, { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import Webcam from "react-webcam";
import {
  Box,
  Button,
  Typography,
  Card,
  CardMedia,
  CardContent,
  CircularProgress,
  Alert,
  Container,
  Paper,
  Divider,
} from "@mui/material";
import CameraAltIcon from "@mui/icons-material/CameraAlt";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ErrorIcon from "@mui/icons-material/Error";

const App = () => {
  const [img1, setImg1] = useState(null);
  const [img2, setImg2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const webcamRef = useRef(null);

  const handleFileChange = (e, setImage) => {
    const file = e.target.files[0];
    if (file) setImage(file);
  };

  const captureImage = useCallback((setImage) => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "captured_image.jpg", {
            type: "image/jpeg",
          });
          setImage(file);
        });
    }
  }, []);

  const handleUpload = async () => {
    if (!img1 || !img2) {
      setError("Please select or capture both images.");
      return;
    }

    setError("");
    setResult(null); // Clear previous result ONLY when starting new verification
    setLoading(true);

    const formData = new FormData();
    formData.append("img1", img1);
    formData.append("img2", img2);

    try {
      const response = await axios.post("http://127.0.0.1:3015/verify/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data.result);
    } catch (err) {
      setError("Error verifying images. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let img1URL, img2URL;
    if (img1) img1URL = URL.createObjectURL(img1);
    if (img2) img2URL = URL.createObjectURL(img2);

    return () => {
      if (img1URL) URL.revokeObjectURL(img1URL);
      if (img2URL) URL.revokeObjectURL(img2URL);
    };
  }, [img1, img2]);

  const confidenceScore = result
    ? Math.max(0, (1 - result.distance / result.threshold) * 100).toFixed(2)
    : null;

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ padding: 4, marginTop: 4, textAlign: "center", borderRadius: 3 }}>
        <Typography variant="h4" fontWeight="bold" gutterBottom>
          Face Verification System
        </Typography>
        <Typography variant="body1" color="text.secondary" gutterBottom>
          Upload or capture images to verify facial similarity.
        </Typography>

        {/* File Upload Section */}
        <Box sx={{ display: "flex", justifyContent: "center", gap: 2, marginBottom: 3 }}>
          <Button variant="contained" component="label" color="secondary" startIcon={<UploadFileIcon />}>
            Upload Image 1
            <input type="file" accept="image/*" hidden onChange={(e) => handleFileChange(e, setImg1)} />
          </Button>
          <Button variant="contained" component="label" color="secondary" startIcon={<UploadFileIcon />}>
            Upload Image 2
            <input type="file" accept="image/*" hidden onChange={(e) => handleFileChange(e, setImg2)} />
          </Button>
        </Box>

        {/* Webcam Section */}
        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, marginBottom: 3 }}>
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            style={{ width: "100%", maxWidth: "350px", borderRadius: "12px", border: "2px solid #ccc" }}
          />
          <Box sx={{ display: "flex", gap: 2 }}>
            <Button variant="contained" color="primary" startIcon={<CameraAltIcon />} onClick={() => captureImage(setImg1)}>
              Capture Image 1
            </Button>
            <Button variant="contained" color="primary" startIcon={<CameraAltIcon />} onClick={() => captureImage(setImg2)}>
              Capture Image 2
            </Button>
          </Box>
        </Box>

        {/* Image Preview */}
        <Box sx={{ display: "flex", justifyContent: "center", gap: 2, marginBottom: 3 }}>
          {img1 && (
            <Card sx={{ maxWidth: 280, borderRadius: 2 }}>
              <CardMedia component="img" height="180" image={URL.createObjectURL(img1)} alt="Image 1" />
              <CardContent>
                <Typography variant="body2" color="text.secondary">Image 1</Typography>
              </CardContent>
            </Card>
          )}
          {img2 && (
            <Card sx={{ maxWidth: 280, borderRadius: 2 }}>
              <CardMedia component="img" height="180" image={URL.createObjectURL(img2)} alt="Image 2" />
              <CardContent>
                <Typography variant="body2" color="text.secondary">Image 2</Typography>
              </CardContent>
            </Card>
          )}
        </Box>

        {/* Verification Button */}
        <Button variant="contained" color="success" onClick={handleUpload} disabled={loading} sx={{ fontSize: "16px", padding: "10px 20px" }}>
          {loading ? "Processing..." : "Verify Faces"}
        </Button>

        {/* Error Message */}
        {error && <Alert severity="error" sx={{ marginTop: 2 }}><ErrorIcon /> {error}</Alert>}

        {/* Results */}
        {result && (
          <Paper elevation={3} sx={{ marginTop: 4, padding: 3, borderRadius: 3 }}>
            <Typography variant="h5" fontWeight="bold" sx={{ marginBottom: 2 }}>
              Verification Result
            </Typography>
            <Typography variant="h6" color={result.verified ? "success.main" : "error.main"}>
              {result.verified ? <CheckCircleIcon /> : <ErrorIcon />} {result.verified ? "Match Found ✅" : "No Match ❌"}
            </Typography>
            <Divider sx={{ marginY: 2 }} />
            {/* <Typography><strong>Distance:</strong> {result.distance.toFixed(4)}</Typography>
            <Typography><strong>Threshold:</strong> {result.threshold}</Typography>
            <Typography><strong>Confidence Score:</strong> {confidenceScore}%</Typography>
            <Typography><strong>Model:</strong> {result.model}</Typography>
            <Typography><strong>Detector Backend:</strong> {result.detector_backend}</Typography>
            <Typography><strong>Similarity Metric:</strong> {result.similarity_metric}</Typography> */}
            {result.time && <Typography><strong>Processing Time:</strong> {result.time.toFixed(2)} sec</Typography>}
          </Paper>
        )}
      </Paper>
    </Container>
  );
};

export default App;