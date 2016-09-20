module l96
! There is also a set of ensemble integrations.
  implicit none
! f2py -c l96_for.f90 -m l96_for only: tinteg1scl emtinteg1scl : (for intel fortran compiler: --fcompiler=intelem)

! global variables of the module
  real(8) :: dt=0.002d0 ! one/two scale time step
!  two scale constants
  integer  :: ktssls=2 !number of small-scale steps per large scale step
  real(8) :: dtss ! derived dt/ktssls
  real(8) :: fss  !derived factor=hint*css/bss
  real(8) :: hint=1.0d0
  real(8) :: css=10.0d0
  real(8) :: bss=10.0d0
  real(8), parameter :: o0ss=1.0, o1ss=0.0 !for shear (o1ss may be a factor)
  integer, parameter :: jparnoloc=0! local/non local polynomial forcing 
                                   ! g(1-->x_n-1,0-->x_n,2-->x_n+1)

contains
!
!--------------------------------------------------------------
!
  SUBROUTINE lorenz96core(nx,ny,xin,xout,force)

! Lorenz 96 model for the slow variables 

    integer, intent(in) :: nx,ny
    REAL(8),INTENT(IN) :: xin(nx),force(ny)
    REAL(8),INTENT(OUT) :: xout(nx)
    INTEGER :: i


!    nx = size(xin,1)
    xout(1) = xin(nx) * ( xin(2) - xin(nx-1) ) - xin(1) + force(1) 
    xout(2) = xin(1) * ( xin(3) - xin(nx) ) - xin(2) + force(2) 
    DO i=3,nx-1
       xout(i) = xin(i-1) * ( xin(i+1) - xin(i-2) ) - xin(i) + force(i)
    END DO
    xout(nx) = xin(nx-1) * ( xin(1) - xin(nx-2) ) - xin(nx) + force(nx)
  
    xout(:) = dt * xout(:)

    RETURN
  END SUBROUTINE lorenz96core
!
!------------------------------------------------------
!
  subroutine tintegss(xin,y,ssforce)

! Time integration of the small-scale model

    implicit none
!  Inputs ktss,fss

    real(8) :: y(:)
    real(8), intent(in)  :: xin(:)
    real(8), intent(out) :: ssforce(:)
    real(8) :: fss
    integer :: k,ny,nx,nj,ix


    nx = size(xin,1)
    ny = size(y,1)
    nj = int(ny/nx)      ! number of small scale points for each large scale point (nxss/nx)


!>>>>> TIME INTEGRATION START
    do k=1,ktssls
       call rk4(lorenz96ss,y,xin)
!       call rk4ss(ny,nx,y,xin)
    end do
!<<<<< TIME INTEGRATION END

    do ix=1,nx
       ssforce(ix)=-fss*sum(y((ix-1)*nj+1:ix*nj))
    enddo

    return
  end subroutine tintegss
!
!-----------------------------------------------------------------------
!
  subroutine lorenz96ss(nxss,nx,yin,yout,xin)

! Lorenz small-scale equations
!  Input parameters: css,bss,hint
!  Cyclic conditions are imposed at the extremes 

!  Notation taken from Wilks QJ 2005.
    integer, intent(in) :: nxss,nx
    real(8),intent(in) :: yin(nxss),xin(nx)
    real(8),intent(out) :: yout(nxss)

    integer :: nj, i,ix,j
    real(8) :: ass,xss
!Eliminar o0ss y o1ss?

! nxss=nj*nx
!    nxss=size(yin,1)
!    nx=size(xin,1)
    nj = int(nxss/nx)
    ass=css*bss 
    
! ix = 1 
!    xss=fss*xin(1)
    xss=fss*(o0ss*xin(1)+o1ss*(xin(2)-xin(nx)))

    yout(1) = ass * yin(2) * ( yin(nxss) - yin(3) ) - css * yin(1) + xss
    DO i=2,nj
       yout(i) = ass * yin(i+1) * ( yin(i-1) - yin(i+2) ) - css * yin(i) + xss
    ENDDO
   
! ix = 2:nx-1
    i=nj
    do ix=2,nx-1
!       xss=fss*xin(ix)
       xss=fss*(o0ss*xin(ix)+o1ss*(xin(ix+1)-xin(ix-1)))

       do j=1,nj
          i=i+1
          yout(i) = ass * yin(i+1) * ( yin(i-1) - yin(i+2) ) - css* yin(i) + xss
       enddo
    enddo

!  ix = nx
!    xss=fss*xin(nx)
    xss=fss*(o0ss*xin(nx)+o1ss*(xin(1)-xin(nx-1)))

    do i=nj*(nx-1)+1,nxss-2 !nxss=nj*nx
       yout(i) = ass *yin(i+1) * ( yin(i-1) - yin(i+2) ) - css* yin(i) + xss
    end do
    yout(nxss-1) = ass * yin(nxss) * ( yin(nxss-2) - yin(1) ) - css*yin(nxss-1) + xss
    yout(nxss) = ass * yin(1) * ( yin(nxss-1) - yin(2) ) - css*yin(nxss) + xss
    
    yout(:) = dtss * yout(:)

    return
  end subroutine lorenz96ss

  subroutine forcing(x0,xpar,mforce)

! Determine forcing for polinomial parameterization
!   Includes local forcing and derivative forcing
    real(8),intent(in) :: x0(:),xpar(:)
    real(8),intent(out) :: mforce(:)
    integer idg,npdim,nxdim,ix

    npdim=size(xpar,1)
    nxdim=size(mforce,1)

    mforce(:)=0.0

    option: select case (jparnoloc)

    case (0)
       do idg=1,npdim
          mforce(:)=mforce(:)+xpar(idg)*x0(:)**(idg-1)
       enddo
     
! forcing depends on the derivative
    case (1)
       do ix=2,nxdim-1
          do idg=1,npdim
             mforce(ix)=mforce(ix)+xpar(idg)*(x0(ix+1)-x0(ix-1))**(idg-1)
          enddo
       enddo
       do idg=1,npdim
          mforce(nxdim)=mforce(nxdim)+xpar(idg)*(x0(1)-x0(nxdim-1))**(idg-1)
          mforce(1)=mforce(1)+xpar(idg)*(x0(2)-x0(nxdim))**(idg-1)
       enddo
       
!    case (2)
!       do ix=1,nxdim
!          do idg=1,npdim
!             mforce(ix)=mforce(ix)+xpar(idg)*x(ix+1)**(idg-1)
!          enddo
!       enddo
    end select option

  end subroutine forcing
!
!-----------------------------------------------------------------------
!
  subroutine rk4(fn,x,force)
!    4th order Runge-Kutta algoritm

    external :: fn
! aparentemente esto no lo entiende el f2py 
    real(8) :: x(:)
    real(8), intent(in) :: force(:)
    real(8), dimension(size(x,1)) :: xtmp,xf1,xf2,xf3,xf4
    integer :: nx,ny
    
    nx=size(x,1)
    ny=size(force,1)

    xtmp = x
    CALL fn(nx,ny,xtmp,xf1,force)
    xtmp = x + 0.5d0 * xf1
    CALL fn(nx,ny,xtmp,xf2,force)
    xtmp = x + 0.5d0 * xf2
    CALL fn(nx,ny,xtmp,xf3,force)
    xtmp = x + xf3
    CALL fn(nx,ny,xtmp,xf4,force)

    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0
 
  end subroutine rk4
!!
!!-----------------------------------------------------------------------
!!
!  subroutine rk4ls(nx,ny,x,force)
!!    4th order Runge-Kutta algoritm
!
!    integer, intent(in) :: nx,ny
!    real(8) :: x(nx)
!    real(8), intent(in) :: force(ny)
!    real(8), dimension(nx) :: xtmp,xf1,xf2,xf3,xf4
!
!    xtmp = x
!    CALL lorenz96core(xtmp,xf1,force)
!    xtmp = x + 0.5d0 * xf1
!    CALL lorenz96core(xtmp,xf2,force)
!    xtmp = x + 0.5d0 * xf2
!    CALL lorenz96core(xtmp,xf3,force)
!    xtmp = x + xf3
!    CALL lorenz96core(xtmp,xf4,force)
!
!    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0
!
!  end subroutine rk4ls
!!
!!-----------------------------------------------------------------------
!!
!  subroutine rk4ss(nx,ny,x,force)
!!    4th order Runge-Kutta algoritm
!
!    integer, intent(in) :: nx,ny
!    real(8) :: x(nx)
!    real(8), intent(in) :: force(ny)
!    real(8), dimension(nx) :: xtmp,xf1,xf2,xf3,xf4
!
!    xtmp = x
!    CALL lorenz96ss(xtmp,xf1,force)
!    xtmp = x + 0.5d0 * xf1
!    CALL lorenz96ss(xtmp,xf2,force)
!    xtmp = x + 0.5d0 * xf2
!    CALL lorenz96ss(xtmp,xf3,force)
!    xtmp = x + xf3
!    CALL lorenz96ss(xtmp,xf4,force)
!
!    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0
!
!  end subroutine rk4ss
!
!------------------------------------------------------
!
  subroutine emtintegss(xin,y,ssforce)

! Time integration of the small-scale model

    implicit none
!  Inputs ktss,fss

    real(8), intent(inout)  :: y(:,:)
    real(8), intent(in)  :: xin(:,:)
    real(8), intent(out) :: ssforce(:,:)
    integer :: k,ny,nx,nj,ix,nem


    nx = size(xin,1)
    ny = size(y,1)
    nem = size(xin, 2)
    nj = int(ny/nx)      ! number of small scale points for each large scale point (nxss/nx)


!>>>>> TIME INTEGRATION START
    do k=1,ktssls
       call emrk4(emlorenz96ss,y,xin)
!       call emrk4ss(ny,nx,nem,y,xin)
    end do
!<<<<< TIME INTEGRATION END

    do ix=1,nx
       ssforce(ix,:)=-fss*sum(y((ix-1)*nj+1:ix*nj,:))
    enddo

    return
  end subroutine emtintegss
!
!-----------------------------------------------------------------------
!
  subroutine emlorenz96ss(yin,yout,xin)

! Lorenz small-scale equations
!  Input parameters: css,bss,hint
!  Cyclic conditions are imposed at the extremes 

!  Notation taken from Wilks QJ 2005.
    real(8),intent(in) :: yin(:,:),xin(:,:)
    real(8),intent(out) :: yout(:,:)

    integer :: nx,nxss
    integer :: nj, i,ix,j
    real(8) :: ass,xss(size(xin,2))
!Eliminar o0ss y o1ss?

! nxss=nj*nx
    nxss=size(yin,1)
    nx=size(xin,1)
    nj = int(nxss/nx)
    ass=css*bss 
    
! ix = 1 
!    xss=fss*xin(1)
    xss(:)=fss*(o0ss*xin(1,:)+o1ss*(xin(2,:)-xin(nx,:)))

    yout(1,:) = ass * yin(2,:) * ( yin(nxss,:) - yin(3,:) ) - css * yin(1,:) + xss(:)
    DO i=2,nj
       yout(i,:) = ass * yin(i+1,:) * ( yin(i-1,:) - yin(i+2,:) ) - css * yin(i,:) + xss(:)
    ENDDO
   
! ix = 2:nx-1
    i=nj
    do ix=2,nx-1
!       xss=fss*xin(ix)
       xss(:)=fss*(o0ss*xin(ix,:)+o1ss*(xin(ix+1,:)-xin(ix-1,:)))

       do j=1,nj
          i=i+1
          yout(i,:) = ass * yin(i+1,:) * ( yin(i-1,:) - yin(i+2,:) ) - css* yin(i,:) + xss(:)
       enddo
    enddo

!  ix = nx
!    xss=fss*xin(nx)
    xss(:)=fss*(o0ss*xin(nx,:)+o1ss*(xin(1,:)-xin(nx-1,:)))

    do i=nj*(nx-1)+1,nxss-2 !nxss=nj*nx
       yout(i,:) = ass *yin(i+1,:) * ( yin(i-1,:) - yin(i+2,:) ) - css* yin(i,:) + xss(:)
    end do
    yout(nxss-1,:) = ass * yin(nxss,:) * ( yin(nxss-2,:) - yin(1,:) ) - css*yin(nxss-1,:) + xss(:)
    yout(nxss,:) = ass * yin(1,:) * ( yin(nxss-1,:) - yin(2,:) ) - css*yin(nxss,:) + xss(:)
    
    yout(:,:) = dtss * yout(:,:)

    return
  end subroutine emlorenz96ss

!
!--------------------------------------------------------------
!
  SUBROUTINE emlorenz96core(xin,xout,force)

! Lorenz 96 model for the slow variables 
!  for ensamble integration (nothing changes)
    REAL(8),INTENT(IN) :: xin(:,:),force(:,:)
    REAL(8),INTENT(OUT) :: xout(:,:)
    INTEGER :: nx,i
    
    nx = size(xin,1)

    xout(1,:) = xin(nx,:) * ( xin(2,:) - xin(nx-1,:) ) - xin(1,:) + force(1,:) 
    xout(2,:) = xin(1,:) * ( xin(3,:) - xin(nx,:) ) - xin(2,:) + force(2,:) 
    DO i=3,nx-1
       xout(i,:) = xin(i-1,:) * ( xin(i+1,:) - xin(i-2,:) ) - xin(i,:) + force(i,:)
    END DO
    xout(nx,:) = xin(nx-1,:) * ( xin(1,:) - xin(nx-2,:) ) - xin(nx,:) + force(nx,:)
  
    xout(:,:) = dt * xout(:,:)
  
    RETURN
  END SUBROUTINE emlorenz96core
!
!-----------------------------------------------------------------------
!
  subroutine emforcing(x0,xpar,mforce)

! Determine forcing for polinomial parameterization
!   Includes local forcing and derivative forcing
    real(8),intent(in) :: x0(:,:),xpar(:,:)
    real(8),intent(out) :: mforce(:,:)
    integer idg,npdim,nxdim,ix

    npdim=size(xpar,1)
    nxdim=size(mforce,1)

    mforce(:,:)=0.0

    option: select case (jparnoloc)

    case (0)
       do idg=1,npdim
          do ix=1,nxdim
             mforce(ix,:)=mforce(ix,:)+xpar(idg,:)*x0(ix,:)**(idg-1)
          enddo
       enddo

! forcing depends on the derivative
    case (1)
       do ix=2,nxdim-1
          do idg=1,npdim
             mforce(ix,:)=mforce(ix,:)+xpar(idg,:)*(x0(ix+1,:)-x0(ix-1,:))**(idg-1)
          enddo
       enddo
       do idg=1,npdim
          mforce(nxdim,:)=mforce(nxdim,:)+xpar(idg,:)*(x0(1,:)-x0(nxdim-1,:))**(idg-1)
          mforce(1,:)=mforce(1,:)+xpar(idg,:)*(x0(2,:)-x0(nxdim,:))**(idg-1)
       enddo
       
!    case (2)
!       do ix=1,nxdim
!          do idg=1,npdim
!             mforce(ix)=mforce(ix)+xpar(idg)*x(ix+1)**(idg-1)
!          enddo
!       enddo
    end select option

  end subroutine emforcing
!
!-----------------------------------------------------------------------
!
  subroutine emrk4(fn,x,force)
!  subroutine emrk4(fn,nx,ny,nem,xold,force,x)
!    4th order Runge-Kutta algoritm
!  for ensamble integration (nothing changes)
    external :: fn
! aparentemente esto no lo entiende el f2py 
!    integer,intent(in) :: nx,ny,nem
 
    real(8) :: x(:,:)
    real(8),intent(in) :: force(:,:)
    real(8), dimension(size(x,1),size(x,2)) :: xtmp, xf1, xf2, xf3, xf4 
    integer :: nx,ny,nem

!    real(8), dimension(size(x,1),size(x,2)) :: xtmp
!    real(8), dimension(size(x,1),size(x,2)) :: xf1
!    real(8), dimension(size(x,1),size(x,2)) :: xf2
!    real(8), dimension(size(x,1),size(x,2)) :: xf3
!    real(8), dimension(size(x,1),size(x,2)) :: xf4
    nx=size(x,1)
    ny=size(force,1)
    nem=size(x,2)

    xtmp=x
    CALL fn(nx,ny,nem,xtmp,xf1,force)
    xtmp = x + 0.5d0 * xf1
    CALL fn(nx,ny,nem,xtmp,xf2,force)
    xtmp = x + 0.5d0 * xf2
    CALL fn(nx,ny,nem,xtmp,xf3,force)
    xtmp = x + xf3
    CALL fn(nx,ny,nem,xtmp,xf4,force)

    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0

  end subroutine emrk4
!
!-----------------------------------------------------------------------
!
  subroutine emrk4ls(nx,ny,nem,x,force)
!    4th order Runge-Kutta algoritm
!  for ensamble integration (nothing changes)
    integer :: nx,ny,nem
    real(8) :: x(nx,nem)
    real(8),intent(in) :: force(ny,nem)
    real(8), dimension(nx,nem) :: xtmp, xf1, xf2, xf3, xf4 

    xtmp = x
    CALL emlorenz96core(xtmp,xf1,force)
    xtmp = x + 0.5d0 * xf1
    CALL emlorenz96core(xtmp,xf2,force)
    xtmp = x + 0.5d0 * xf2
    CALL emlorenz96core(xtmp,xf3,force)
    xtmp = x + xf3
    CALL emlorenz96core(xtmp,xf4,force)

    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0

  end subroutine emrk4ls
!
!-----------------------------------------------------------------------
!
  subroutine emrk4ss(nx,ny,nem,x,force)
!    4th order Runge-Kutta algoritm
!  for ensamble integration (nothing changes)
    integer :: nx,ny,nem
    real(8) :: x(nx,nem)
    real(8),intent(in) :: force(ny,nem)
    real(8), dimension(nx,nem) :: xtmp, xf1, xf2, xf3, xf4 
!    real(8), dimension(size(x,1),size(x,2)) :: xtmp
!    real(8), dimension(size(x,1),size(x,2)) :: xf1
!    real(8), dimension(size(x,1),size(x,2)) :: xf2
!    real(8), dimension(size(x,1),size(x,2)) :: xf3
!    real(8), dimension(size(x,1),size(x,2)) :: xf4

    xtmp = x
    CALL emlorenz96ss(xtmp,xf1,force)
    xtmp = x + 0.5d0 * xf1
    CALL emlorenz96ss(xtmp,xf2,force)
    xtmp = x + 0.5d0 * xf2
    CALL emlorenz96ss(xtmp,xf3,force)
    xtmp = x + xf3
    CALL emlorenz96ss(xtmp,xf4,force)

    x = x + ( xf1 + 2.0d0 * xf2 + 2.0d0 * xf3 + xf4 ) / 6.0d0

  end subroutine emrk4ss
!
!-----------------------------------------------------------------------
!
  SUBROUTINE tinteg2scl(nx,nss,kt,xin,xssin,mforce,tmssforce,hint1,css1,bss1,dt1,x,xss)

! Time integration lorensz 96 2-scale with rk4

! input nx,nxss,kt,x,hint,css,bss
! output x, tmssforce
!    use l96
    integer :: nx,nss,kt
    real(8),dimension(nx),intent(in) :: xin !inout (quizas convenga definirlo como salida tb xin,xout?
    real(8),dimension(nss),intent(in) :: xssin !inout
    real(8),dimension(nx), intent(in) :: mforce
    real(8), intent(in) :: hint1,css1,bss1,dt1

    real(8),dimension(nx), intent(out) ::x
    real(8),dimension(nss), intent(out) ::xss
    real(8),dimension(nx), intent(out) ::tmssforce

    real(8),dimension(nx) :: ssforce,ssforce1
!    real(8) :: fss
    integer :: k,ix
    integer :: nj

! initializes global variables
    dt=dt1
    hint=hint1
    css=css1
    bss=bss1
    fss=hint*css/bss
    dtss=dt/ktssls

! input variables
    x=xin
    xss=xssin

    nj = int(nss/nx)      ! number of small scale points for each large scale point (nss/nx)


!  small-scale forcing of the initial condition
    DO ix=1,nx
       ssforce(ix)=-fss*sum(xss((ix-1)*nj+1:ix*nj))
    ENDDO

    tmssforce=0.0

!>>>>> TIME INTEGRATION START
    DO k=1,kt


       call tintegss(x,xss,ssforce1) !with old x

       ssforce=mforce+ssforce
       call rk4(lorenz96core,x,ssforce)!dt implicitly
!       call rk4ls(nx,nx,x,ssforce)!dt implicitly

       tmssforce(:)=tmssforce(:)+ssforce(:)
       ssforce=ssforce1

    END DO
!<<<<< TIME INTEGRATION END

    tmssforce=tmssforce/kt

    RETURN
  END SUBROUTINE tinteg2scl


!
!-----------------------------------------------------------------------
!
  SUBROUTINE tinteg1scl(nx,npar,kt,xin,xpar,dt1,x)

! Time integration slow variables with rk4
!   Perfect and imperfect model (2-scales 1-scale lorenz 96)
!  m%x includes both large and small scale in 2-scales

! input nx,nxss,kt,x
! output x, tmssforce
!    use l96
 
    integer :: nx,npar,kt
    real(8), dimension(nx), intent(in) :: xin
    real(8), dimension(nx), intent(out) :: x
    real(8), dimension(npar), intent(in) :: xpar! needed for openmp??
    real(8), intent(in) :: dt1


    real(8), dimension(nx) :: force
    logical  :: lpoforcing ! polynomial forcing
    integer :: k

    dt=dt1
    
    x=xin
!  with constant or polynomial forcing
    lpoforcing=.FALSE. 
    if (npar .eq. nx) then
       force(:)=xpar(:)
    else
       lpoforcing=.TRUE.  ! polynomial forcing
    endif

!>>>>> TIME INTEGRATION START
    DO k=1,kt
       if (lpoforcing) call forcing(x,xpar,force) !polynomial forcing evolves with x
       call rk4(lorenz96core,x,force)!dt implicitly
!       call rk4ls(nx,nx,x,force)!dt implicitly

    END DO
!<<<<< TIME INTEGRATION END



    RETURN
  END SUBROUTINE tinteg1scl
!-----------------------------------------------------------------------
!
!  Ensemble integrations
!
!-----------------------------------------------------------------------
!
! There is no difference between the codes except the extra dimension
!-----------------------------------------------------------------------
!
  SUBROUTINE emtinteg2scl(nx,nss,nem,kt,xin,xssin,mforce,hint1,css1,bss1,dt1,x,xss)

! Time integration lorensz 96 2-scale with rk4

! input nx,nxss,kt,x,hint,css,bss
! output x, tmssforce
!    use l96
 
    integer :: nx,nss,nem,kt
    real(8),dimension(nx,nem),intent(in) :: xin 
    real(8),dimension(nss,nem),intent(in) :: xssin
    real(8),dimension(nx,nem), intent(in) :: mforce
    real(8), intent(in) :: hint1,css1,bss1,dt1

    real(8),dimension(nx,nem),intent(out) :: x
    real(8),dimension(nss,nem),intent(out) :: xss

    real(8),dimension(nx,nem) :: ssforce,ssforce1

!    real(8) :: fss
    integer :: k,ix
    integer :: nj

! initializes global variables
    dt=dt1
    hint=hint1
    css=css1
    bss=bss1
    fss=hint*css/bss
    dtss=dt/ktssls


    nj = int(nss/nx)      ! number of small scale points for each large scale point (nss/nx)

    x=xin
    xss=xssin


!  small-scale forcing of the initial condition
    DO ix=1,nx
       ssforce(ix,:)=-fss*sum(xss((ix-1)*nj+1:ix*nj,:))
    ENDDO

!>>>>> TIME INTEGRATION START
    DO k=1,kt

       call emtintegss(x,xss,ssforce1) !with old x

       ssforce=mforce+ssforce
       call emrk4(emlorenz96core,x,ssforce)!dt implicitly
!       call emrk4ls(nx,nx,nem,x,ssforce)!dt implicitly

       ssforce=ssforce1

    END DO
!<<<<< TIME INTEGRATION END


    RETURN
  END SUBROUTINE emtinteg2scl


!
!-----------------------------------------------------------------------
!
  SUBROUTINE emtinteg1scl(nx,npar,nem,kt,xin,xpar,dt1,x)

! Time integration slow variables with rk4
!    use l96
 
    integer :: nx,npar,nem,kt
    real(8), dimension(nx,nem),intent(in) :: xin
    real(8), dimension(npar,nem), intent(in) :: xpar
    real(8), intent(in) :: dt1

    real(8), dimension(nx,nem),intent(out) :: x
    real(8), dimension(nx,nem) :: force
    logical  :: lpoforcing ! polynomial forcing
    integer :: k

    dt=dt1
    x=xin

!  with constant or polynomial forcing
    if (npar .eq. nx) then
       force(:,:)=xpar(:,:)
    else
       lpoforcing=.TRUE.  ! polynomial forcing
    endif

!>>>>> TIME INTEGRATION START
    DO k=1,kt

       if (lpoforcing) call emforcing(x,xpar,force) !polynomial forcing evolves with x
       call emrk4(emlorenz96core,x,force)!dt implicitly
!       call emrk4ls(nx,nx,nem,x,force)!dt implicitly

    END DO
!<<<<< TIME INTEGRATION END



    RETURN
  END SUBROUTINE emtinteg1scl

end module l96
