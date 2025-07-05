import React from 'react';



export default function Chatbot() {
  return (
    <div className="chatbot">
      <div className="chat-main">
        <h2 className="chat-title">Dental Assistant</h2>
      </div>

      <div className="user-profile">
        <div className="user-user" id="user-avatar">U</div>
        <div className="user-info">
          <div className="user-user" id="user-welcome" >Welcome back,</div>
          <div className="user-user" id="user-name">User</div>
        </div>
      </div>

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <input
            type="text"
            placeholder="Type a new message here"
            className="chat-input"
          />
          <button className="input-action-button">Submit</button>
        </div>
      </div>













      {/* Chatting Area
      <div className="section">
        <h2>Chat Assistant</h2>
        <div className="chat-section">
          <div className="chat-messages">

          </div>
        
          <div className="chat-input">
            <input 
              type="text"
              placeholder="Ask about dental procedures, diagnosis, or treatment..." />

            <button className="chatbot-send-button">Send</button>
          </div>
        </div>
        */}

        {/* Chat Header 
        <div id="chat-header">
          <h2>Chat Assist</h2>
        </div>
      
        <div className="chat-messages">
          <div></div>
        </div>

        <div className="chat-input">
          <input type="text" placeholder="Type your message here..." />
          <button className="chatbot-send-button">Send</button>
        </div>
        */}

      

      {/* Upload Area 
      <div className="upload-container">
        <div className="header">
          <h2>File Upload</h2>
          <p>Upload files to share with the chatbot</p>
        </div>
        
        <div className="upload-area" id="upload-area">
          <div className="upload-icon">☁️</div>
          <div class="upload-text">Drag & drop X-ray images here or click to browse</div>
          
          <button class="browse-btn" onclick="document.getElementById('fileInput').click()">Browse Files</button>
          <input type="file" id="fileInput" class="file-input" accept="image/*" />

          <button className="upload-button">Upload</button>
        </div>
      </div>
        */}
    


    </div>
  );
}

