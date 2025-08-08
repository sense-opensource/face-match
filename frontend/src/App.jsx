import { useState, useRef } from 'react'
import {
  Container,
  Box,
  Paper,
  Typography,
  Divider,
  Button,
  CircularProgress,
  Alert,
  Snackbar,
  Avatar
} from '@mui/material'
import CameraAltIcon from '@mui/icons-material/CameraAlt'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ErrorIcon from '@mui/icons-material/Error'
import FaceIcon from '@mui/icons-material/Face'
import SecurityIcon from '@mui/icons-material/Security'
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser'
import axios from 'axios'
import Webcam from 'react-webcam'
import './App.css'

// API URL - change if your backend is on a different port
const API_URL = 'http://localhost:3015'

function App() {
  // State for images
  const [idCardImage, setIdCardImage] = useState(null)
  const [selfieImage, setSelfieImage] = useState(null)
  
  // State for camera
  const [showIdCamera, setShowIdCamera] = useState(false)
  const [showSelfieCamera, setShowSelfieCamera] = useState(false)
  const webcamRef = useRef(null)
  
  // State for verification
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showError, setShowError] = useState(false)

  // Handler for file upload
  const handleFileUpload = (event, setImage) => {
    const file = event.target.files[0]
    if (file) {
      setImage(file)
    }
  }

  // Handler for webcam capture
  const handleCapture = (setImage, closeCamera) => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot()
      // Convert base64 to file
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], 'captured-image.jpg', { type: 'image/jpeg' })
          setImage(file)
          closeCamera()
        })
    }
  }

  // Handler for verification
  const handleVerify = async () => {
    if (!idCardImage || !selfieImage) {
      setError('Please provide both ID card and selfie images')
      setShowError(true)
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('id_card', idCardImage)
      formData.append('photo', selfieImage)

      const response = await axios.post(`${API_URL}/face-match`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      setResult(response.data)
    } catch (err) {
      console.error('Verification error:', err)
      setError(err.response?.data?.detail || 'Verification failed. Please try again.')
      setShowError(true)
    } finally {
      setLoading(false)
    }
  }

  // Handler for reset
  const handleReset = () => {
    setIdCardImage(null)
    setSelfieImage(null)
    setResult(null)
    setError('')
  }

  // Calculate confidence score
  const confidenceScore = result
    ? Math.max(0, Math.min(100, (1 - result.distance / result.threshold) * 100)).toFixed(1)
    : 0

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      {/* Header with Logo */}
      <Paper 
        elevation={2} 
        sx={{ 
          p: 3, 
          mb: 4, 
          borderRadius: 3,
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
          {/* Logo Option 1: Using Material-UI Icons */}
          <Avatar
            sx={{
              width: 60,
              height: 60,
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              mr: 2,
              border: '2px solid rgba(255, 255, 255, 0.3)'
            }}
          >
            <VerifiedUserIcon sx={{ fontSize: 35, color: 'white' }} />
          </Avatar>
          
          {/* Logo Option 2: Custom Image (uncomment to use) */}
          {/* 
          <img 
            src="/path/to/your/logo.png" 
            alt="Company Logo" 
            style={{
              width: '60px',
              height: '60px',
              marginRight: '16px',
              borderRadius: '50%',
              border: '2px solid rgba(255, 255, 255, 0.3)'
            }}
          />
          */}
          
          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h4" component="h1" fontWeight="bold" sx={{ mb: 1 }}>
              Sense Verification
            </Typography>
            <Typography variant="subtitle1" sx={{ opacity: 0.9, fontWeight: 300 }}>
              Advanced AI-Powered Face Verification System
            </Typography>
          </Box>
        </Box>
        
        {/* Feature Icons */}
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4, mt: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityIcon sx={{ fontSize: 20 }} />
            <Typography variant="body2">Secure</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <FaceIcon sx={{ fontSize: 20 }} />
            <Typography variant="body2">AI-Powered</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CheckCircleIcon sx={{ fontSize: 20 }} />
            <Typography variant="body2">Accurate</Typography>
          </Box>
        </Box>
      </Paper>

      {/* Image Upload Section */}
      <Paper elevation={3} sx={{ p: 3, mb: 4, borderRadius: 2 }}>
        <Typography variant="h5" gutterBottom>
          Upload or Capture Images
        </Typography>
        <Divider sx={{ mb: 3 }} />
        
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 3 }}>
          {/* ID Card Section */}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              ID Card
            </Typography>
            {idCardImage ? (
              // Show uploaded image
              <Box sx={{ mb: 2 }}>
                <img
                  src={URL.createObjectURL(idCardImage)}
                  alt="ID Card"
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }}
                />
                <Button
                  variant="outlined"
                  color="secondary"
                  size="small"
                  onClick={() => setIdCardImage(null)}
                  sx={{ mt: 1 }}
                >
                  Change
                </Button>
              </Box>
            ) : showIdCamera ? (
              // Show webcam for ID card
              <Box sx={{ mb: 2 }}>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ facingMode: "user" }}
                  style={{
                    width: '100%',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }}
                />
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleCapture(setIdCardImage, () => setShowIdCamera(false))}
                    startIcon={<CameraAltIcon />}
                  >
                    Capture
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => setShowIdCamera(false)}
                  >
                    Cancel
                  </Button>
                </Box>
              </Box>
            ) : (
              // Show upload options
              <Box
                sx={{
                  border: '1px dashed #ccc',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  mb: 2
                }}
              >
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Upload or capture your ID card
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<UploadFileIcon />}
                  >
                    Upload
                    <input
                      type="file"
                      hidden
                      accept="image/*"
                      onChange={(e) => handleFileUpload(e, setIdCardImage)}
                    />
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<CameraAltIcon />}
                    onClick={() => setShowIdCamera(true)}
                  >
                    Camera
                  </Button>
                </Box>
              </Box>
            )}
          </Box>

          {/* Selfie Section */}
          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" gutterBottom>
              Selfie
            </Typography>
            {selfieImage ? (
              // Show uploaded image
              <Box sx={{ mb: 2 }}>
                <img
                  src={URL.createObjectURL(selfieImage)}
                  alt="Selfie"
                  style={{
                    width: '100%',
                    height: 'auto',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }}
                />
                <Button
                  variant="outlined"
                  color="secondary"
                  size="small"
                  onClick={() => setSelfieImage(null)}
                  sx={{ mt: 1 }}
                >
                  Change
                </Button>
              </Box>
            ) : showSelfieCamera ? (
              // Show webcam for selfie
              <Box sx={{ mb: 2 }}>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={{ facingMode: "user" }}
                  style={{
                    width: '100%',
                    borderRadius: '8px',
                    border: '1px solid #ddd'
                  }}
                />
                <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={() => handleCapture(setSelfieImage, () => setShowSelfieCamera(false))}
                    startIcon={<CameraAltIcon />}
                  >
                    Capture
                  </Button>
                  <Button
                    variant="outlined"
                    onClick={() => setShowSelfieCamera(false)}
                  >
                    Cancel
                  </Button>
                </Box>
              </Box>
            ) : (
              // Show upload options
              <Box
                sx={{
                  border: '1px dashed #ccc',
                  borderRadius: 2,
                  p: 3,
                  textAlign: 'center',
                  mb: 2
                }}
              >
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Upload or capture a selfie
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 2 }}>
                  <Button
                    variant="outlined"
                    component="label"
                    startIcon={<UploadFileIcon />}
                  >
                    Upload
                    <input
                      type="file"
                      hidden
                      accept="image/*"
                      onChange={(e) => handleFileUpload(e, setSelfieImage)}
                    />
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<CameraAltIcon />}
                    onClick={() => setShowSelfieCamera(true)}
                  >
                    Camera
                  </Button>
                </Box>
              </Box>
            )}
          </Box>
        </Box>

        {/* Verification Buttons */}
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            disabled={!idCardImage || !selfieImage || loading}
            onClick={handleVerify}
            startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <VerifiedUserIcon />}
            sx={{
              background: 'linear-gradient(45deg, #667eea 30%, #764ba2 90%)',
              '&:hover': {
                background: 'linear-gradient(45deg, #5a6fd8 30%, #6a4190 90%)',
              }
            }}
          >
            {loading ? 'Verifying...' : 'Verify Faces'}
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            size="large"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </Button>
        </Box>
      </Paper>

      {/* Results Section */}
      {result && (
        <Paper elevation={3} sx={{ p: 3, mb: 4, borderRadius: 2 }}>
          <Typography variant="h5" gutterBottom>
            Verification Results
          </Typography>
          <Divider sx={{ mb: 3 }} />

          {/* Verification Status */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              p: 2,
              mb: 3,
              borderRadius: 2,
              backgroundColor: result.verified ? 'success.light' : 'error.light',
              color: result.verified ? 'success.dark' : 'error.dark'
            }}
          >
            {result.verified ? (
              <CheckCircleIcon sx={{ mr: 1, fontSize: 28 }} />
            ) : (
              <ErrorIcon sx={{ mr: 1, fontSize: 28 }} />
            )}
            <Typography variant="h6">
              {result.verified ? 'Verification Successful' : 'Verification Failed'}
            </Typography>
          </Box>

          {/* Confidence Score */}
          {/* <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle1" gutterBottom>
              Confidence Score: {confidenceScore}%
            </Typography>
            <Box
              sx={{
                height: 10,
                width: '100%',
                backgroundColor: '#e0e0e0',
                borderRadius: 5,
                overflow: 'hidden'
              }}
            >
              <Box
                sx={{
                  height: '100%',
                  width: `${confidenceScore}%`,
                  backgroundColor: confidenceScore > 70
                    ? 'success.main'
                    : confidenceScore > 40
                    ? 'warning.main'
                    : 'error.main',
                  borderRadius: 5,
                  transition: 'width 1s ease-in-out'
                }}
              />
            </Box>
          </Box> */}

          {/* Details Table */}
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 2 }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              Details
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #eee' }}>
              <Typography variant="body2" color="text.secondary">Distance</Typography>
              <Typography variant="body2">{result.distance.toFixed(4)}</Typography>
            </Box>
            {result.threshold && (
              <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #eee' }}>
                <Typography variant="body2" color="text.secondary">Threshold</Typography>
                <Typography variant="body2">{result.threshold.toFixed(2)}</Typography>
              </Box>
            )}
            {/* <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: '1px solid #eee' }}>
              <Typography variant="body2" color="text.secondary">Model</Typography>
              <Typography variant="body2">{result.model || 'ArcFace'}</Typography>
            </Box> */}
            {result.processing_time && (
              <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}>
                <Typography variant="body2" color="text.secondary">Processing Time</Typography>
                <Typography variant="body2">{result.processing_time.toFixed(2)} seconds</Typography>
              </Box>
            )}
          </Box>
        </Paper>
      )}

      {/* Error Snackbar */}
      <Snackbar
        open={showError}
        autoHideDuration={6000}
        onClose={() => setShowError(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setShowError(false)}
          severity="error"
          sx={{ width: '100%' }}
        >
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App;