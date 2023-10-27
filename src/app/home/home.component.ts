import { Component, OnInit, ViewChild } from '@angular/core';
import { HomeService } from './services/home.service';
import { FileUpload } from 'primeng/fileupload';

interface UploadEvent {
  originalEvent: Event;
  files: File[];
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css']
})
export class HomeComponent implements OnInit{
  @ViewChild('primeFileUpload')
  primeFileUpload!: FileUpload;
  constructor(private homeService: HomeService) { }

  pdfUploaded: boolean = false;
  question: string = '';
  answer: string = '';
  uploadedFiles: any[] = [];
  ngOnInit(): void {
    this.homeService.getData().subscribe((x) => {
      console.log("OBS",x)
    })
  }
  

  onUpload(event: any) {
    console.log("HERE")
    for(let file of event.files) {
      this.uploadedFiles.push(file);
    }
  }

  askQuestion() {
    this.homeService.sendQuestion(this.question).subscribe((x:any) => {
      this.answer = x
    })
  }
}
