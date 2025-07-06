import React, { useState, useRef } from 'react';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { 
      id: 1,
      sender: 'Assistant', 
      text: 'Hello! I\'m your dental assistant. Upload an x-ray image for prediagnostic evaluation.',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const[isLoading,setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Prevent default drag behavior on the entire page
  React.useEffect(() => {
    const handleGlobalDrag = (e) => {
      e.preventDefault();
    };

    document.addEventListener('dragover', handleGlobalDrag);
    document.addEventListener('drop', handleGlobalDrag);

    return () => {
      document.removeEventListener('dragover', handleGlobalDrag);
      document.removeEventListener('drop', handleGlobalDrag);
    };
  }, []);

  const handleSendMessage = async () => {
    if (!input && !file) return;

    // Add user message to chat
    if(file){
      setMessages(prev => [...prev, { 
        id: Date.now()+0.5,
        sender: 'User', 
        text: `📷 Uploaded image: ${file.name}`,
        image: imagePreview, // Add the image preview to the message
        timestamp: new Date()
      }]);
    }
    if (input) {
      setIsLoading(true);
      setMessages(prev => [...prev, { 
        id: Date.now(),
        sender: 'User', 
        text: input,
        timestamp: new Date()
      }]);
    }

    // Prepare form data
    const formData = new FormData();
    if (file) {
      formData.append('file', file);
    }
    if (input) {
      formData.append('message', input);
    }

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      console.log('Got this back:', data);
    const diagnosticMessage = `The diagnosed patient has ${data.primary_finding || 'unknown condition'} of ${data.severity || 'moderate'} severity. Thus, there is an urgency of ${data.urgency || 'moderate'} sso the next steps I would recommend is ${data.recommendations && data.recommendations.length > 0 ? data.recommendations.join(', ') : 'clinical evaluation'}.`;

    setMessages(prev => [...prev, { 
      id: Date.now() + 1,
      sender: 'Assistant', 
      text: diagnosticMessage,
      timestamp: new Date()
    }]);

      // Add assistant's reply to chat
      if (data.user_message) {
        setMessages(prev => [...prev, { 
          id: Date.now() + 1,
          sender: 'Dental Assistant', 
          text: data.user_message,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setIsLoading(false);
      setMessages(prev => [...prev, { 
        id: Date.now() + 1,
        sender: 'Assistant', 
        text: 'Sorry, there was an error processing your request.',
        timestamp: new Date()
      }]);
    }

    // Clear input + file
    setInput('');
    setFile(null);
    setImagePreview(null);
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      console.log('File selected via input:', {
        name: selectedFile.name,
        type: selectedFile.type,
        size: selectedFile.size
      });
      
      // Check for image files more comprehensively
      const isImage = selectedFile.type.startsWith('image/') || 
                     /\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif)$/i.test(selectedFile.name);
      
      if (isImage) {
        setFile(selectedFile);
        // Create preview
        const reader = new FileReader();
        reader.onload = (e) => {
          console.log('FileReader loaded for input file');
          setImagePreview(e.target.result);
        };
        reader.onerror = (e) => {
          console.error('FileReader error for input file:', e);
        };
        reader.readAsDataURL(selectedFile);
      } else {
        console.log('Invalid file type selected:', selectedFile.name, 'Type:', selectedFile.type);
        alert('Please select an image file (JPG, PNG, GIF, etc.)');
        // Clear the input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    }
  };

  const removeImage = () => {
    setFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag enter - target:', e.target);
    setIsDragOver(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag over - target:', e.target);
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag leave - target:', e.target);
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drop event - target:', e.target);
    console.log('Files dropped:', e.dataTransfer.files);
    console.log('File types:', Array.from(e.dataTransfer.files).map(f => f.type));
    setIsDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    
    if (!droppedFile) {
      console.log('No file dropped');
      return;
    }
    
    console.log('File details:', {
      name: droppedFile.name,
      type: droppedFile.type,
      size: droppedFile.size,
      lastModified: droppedFile.lastModified
    });
    
    // Check for image files more comprehensively
    const isImage = droppedFile.type.startsWith('image/') || 
                   /\.(jpg|jpeg|png|gif|bmp|webp|tiff|tif)$/i.test(droppedFile.name);
    
    if (isImage) {
      console.log('Valid image file dropped:', droppedFile.name, 'Type:', droppedFile.type);
      setFile(droppedFile);
      const reader = new FileReader();
      reader.onload = (e) => {
        console.log('FileReader loaded, setting image preview');
        setImagePreview(e.target.result);
      };
      reader.onerror = (e) => {
        console.error('FileReader error:', e);
      };
      reader.readAsDataURL(droppedFile);
    } else {
      console.log('Invalid file type. File:', droppedFile.name, 'Type:', droppedFile.type);
      alert('Please drop an image file (JPG, PNG, GIF, etc.)');
    }
  };

  const handleDropZoneClick = () => {
    fileInputRef.current?.click();
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-header">
        <h1 className="chatbot-title">Dentura</h1>
        <p className="chatbot-subtitle">The Dental AI Prediagnostic Assistant</p>
      </div>

      <div className="chatbot-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.sender === 'User' ? 'message-user' : 'message-assistant'}`}>
            <div className="message-content">
              <div className="message-text">{msg.text}</div>
              {msg.image && (
                <div className="message-image">
                  <img src={msg.image} alt="Uploaded image" className="chat-image" />
                </div>
              )}
              <div className="message-time">{formatTime(msg.timestamp)}</div>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message message-assistant">
            <div className="message-content">
              <div className="message-text"> Analyzing your X-ray...</div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="chatbot-input-area">
        {imagePreview && (
          <div className="image-preview-container">
            <img src={imagePreview} alt="Preview" className="image-preview" />
            <button onClick={removeImage} className="remove-image-btn">×</button>
          </div>
        )}
        
        <div 
          className={`file-drop-zone ${isDragOver ? 'drag-over' : ''}`}
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleDropZoneClick}
          style={{ userSelect: 'none' }}
        >
          <input 
            ref={fileInputRef}
            type="file"
            className="file-input"
            onChange={handleFileChange}
            accept="image/*,.jpg,.jpeg,.png,.gif,.bmp,.webp,.tiff,.tif"
          />
          <div className="drop-zone-content">
            <div className="upload-icon">📁</div>
            <p className="drop-zone-text">
              {isDragOver ? 'Drop your image here' : 'Drag & drop an image here, or click to browse'}
            </p>
          </div>
        </div>

        <div className="input-container">
          <button
            className="send-button"
            onClick={handleSendMessage}
            disabled={(!input && !file)||isLoading}>
            {isLoading ? 'Analyzing...': 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
